"""Tầng Core: đọc Staging bằng Spark, khử trùng, làm sạch tiếng Việt, ghi Parquet ra data/core/.

Mỗi lần chạy dựng lại toàn bộ Core từ Staging (Staging chỉ ghi thêm, không bao giờ bị sửa) rồi
đổi tên thư mục một lần, nên chạy lại bao nhiêu lần cũng ra cùng kết quả và không bao giờ để lại
bảng dở dang.

Chạy tay:  docker compose --profile debug run --rm airflow-cli python -m ytb.spark_core
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import config, text_clean


def core_dir() -> Path:
    return config.DATA_DIR / "core"


def build_spark(app_name: str = "ytb_core"):
    from pyspark.sql import SparkSession

    # Worker Python của Spark phải dùng đúng Python đang chạy (có pyvi, pyarrow...).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        # Sau này muốn có cluster thì chỉ cần đặt YTB_SPARK_MASTER=spark://spark-master:7077, không sửa code.
        .master(os.getenv("YTB_SPARK_MASTER", "local[2]"))
        .config("spark.driver.memory", os.getenv("YTB_SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _swap_in(tmp: Path, final: Path) -> None:
    """Thay bảng cũ bằng bảng mới vừa dựng xong."""
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def _read_staging(spark, table: str):
    folder = config.STAGING_DIR / table
    if not folder.exists() or not any(folder.rglob("*.parquet")):
        return None
    return spark.read.option("basePath", str(folder)).parquet(str(folder))


def _write_core(df, table: str, partitions: int) -> int:
    final = core_dir() / table
    tmp = core_dir() / "_building" / table
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.repartition(partitions).write.mode("overwrite").option("compression", "zstd").parquet(str(tmp))
    _swap_in(tmp, final)
    return int(df.sparkSession.read.parquet(str(final)).count())


def _ts(col):
    from pyspark.sql import functions as F
    return F.col(col).cast("timestamp")


def run_core(spark=None) -> dict:
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    own_session = spark is None
    spark = spark or build_spark()
    summary: dict = {}
    try:
        # ---------- bình luận ----------
        raw = _read_staging(spark, "comments")
        if raw is not None:
            n_staging = raw.count()
            latest = Window.partitionBy("comment_id").orderBy(
                F.col("updated_at").desc_nulls_last(), F.col("fetched_at").desc_nulls_last())
            dedup = (raw.filter(F.col("comment_id").isNotNull())
                     .withColumn("_rn", F.row_number().over(latest))
                     .filter("_rn = 1").drop("_rn"))

            vstg = _read_staging(spark, "videos")
            if vstg is not None and "is_ai_music" in vstg.columns:
                relevant = vstg.filter(F.col("is_ai_music")).select("video_id").distinct()
                dedup = dedup.join(relevant, "video_id", "left_semi")   # chỉ giữ bình luận của video nhạc AI
                
            out_type = T.StructType([
                T.StructField("text_clean", T.StringType()),
                T.StructField("tokens", T.ArrayType(T.StringType())),
            ])
            process = F.udf(lambda s: text_clean.process(s), out_type)

            core = (dedup
                    .withColumn("_p", process(F.col("text")))
                    .select(
                        "comment_id", "video_id", "thread_id", "parent_id", "is_reply",
                        "author_channel_id",   # không giữ tên hiển thị ở Core: chỉ cần để đếm số người bình luận
                        F.col("text").alias("text_raw"),
                        F.col("_p.text_clean").alias("text_clean"),
                        F.col("_p.tokens").alias("tokens"),
                        F.size("_p.tokens").alias("n_tokens"),
                        "like_count", "total_reply_count", "source",
                        _ts("published_at").alias("published_at"),
                        _ts("updated_at").alias("updated_at"),
                        F.to_date(_ts("published_at")).alias("published_date"),
                    )
                    .withColumn("is_modelable", F.col("n_tokens") >= text_clean.MIN_TOKENS)
                    .withColumn("processed_at", F.current_timestamp()))

            n_core = _write_core(core, "comments", partitions=4)
            stats = spark.read.parquet(str(core_dir() / "comments")).agg(
                F.sum(F.when(F.col("text_clean") == "", 1).otherwise(0)).alias("empty"),
                F.sum(F.col("is_modelable").cast("int")).alias("modelable"),
            ).first()
            summary["comments"] = {
                "staging_rows": n_staging, "core_rows": n_core,
                "duplicates_removed": n_staging - n_core,
                "empty_after_cleaning": int(stats["empty"] or 0),
                "modelable": int(stats["modelable"] or 0),
            }

        # ---------- video ----------
        raw = _read_staging(spark, "videos")
        if raw is not None:
            pick = Window.partitionBy("video_id").orderBy(
                (F.col("source") == "api").desc(), F.col("fetched_at").desc_nulls_last())
            videos = (raw.filter(F.col("video_id").isNotNull())
                      .withColumn("_rn", F.row_number().over(pick)).filter("_rn = 1").drop("_rn")
                      .withColumn("published_at", _ts("published_at"))
                      .drop("ingest_date"))
            summary["videos"] = {"core_rows": _write_core(videos, "videos", partitions=1)}

        # ---------- snapshot (mỗi video một dòng mỗi ngày) ----------
        raw = _read_staging(spark, "video_snapshots")
        if raw is not None:
            pick = Window.partitionBy("video_id", "snapshot_date").orderBy(F.col("snapshot_at").desc())
            snaps = (raw.withColumn("_rn", F.row_number().over(pick)).filter("_rn = 1").drop("_rn")
                     .withColumn("snapshot_date", F.to_date("snapshot_date"))
                     .withColumn("snapshot_at", _ts("snapshot_at"))
                     .drop("ingest_date"))
            summary["video_snapshots"] = {"core_rows": _write_core(snaps, "video_snapshots", partitions=1)}
    finally:
        if own_session:
            spark.stop()
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_core(), ensure_ascii=False, indent=2))