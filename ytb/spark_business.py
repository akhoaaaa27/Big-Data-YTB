
"""Business layer: Core -> Silver -> GSDMM/aspect/sentiment -> Business."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from . import config


def build_silver() -> Path:
    """Tạo snapshot Silver mới nhất từ Core; Core không bị sửa.

    Ghi ra file tạm rồi thay thế atomically để retry hằng ngày không bị chặn
    bởi output của lần chạy trước và không để lại file Parquet ghi dở.
    """
    import pyarrow.parquet as pq

    core_dir = config.DATA_DIR / "core" / "comments"
    output_dir = config.DATA_DIR / "silver" / "comments_clean"
    output_file = output_dir / "comments_clean.parquet"

    files = sorted(core_dir.rglob("*.parquet")) if core_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy Core Parquet trong {core_dir}. "
            "Hãy chạy spark_core trước."
        )

    frames = [pq.read_table(file).to_pandas() for file in files]
    df = pd.concat(frames, ignore_index=True)

    required = {"comment_id", "video_id", "text_raw", "tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Core comments thiếu cột: {sorted(missing)}")

    # Giữ comment ID hợp lệ, văn bản gốc và token đã xử lý.
    df = df.dropna(subset=["comment_id", "video_id", "text_raw"]).copy()
    df["comment_id"] = df["comment_id"].astype(str).str.strip()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["text"] = df["text_raw"].astype(str).str.strip()

    df = df[
        df["comment_id"].ne("")
        & df["video_id"].ne("")
        & df["text"].ne("")
    ].copy()

    # Giữ một bản ghi cho mỗi comment_id.
    df = df.drop_duplicates(subset=["comment_id"], keep="last")

    # Repo GSDMM dùng Silver đã bỏ reply; thống nhất chính sách tại đây.
    if "is_reply" in df.columns:
        df = df[~df["is_reply"].fillna(False).astype(bool)].copy()

    # Chuẩn hóa tokens thành list[str].
    def normalize_tokens(value):
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
        if hasattr(value, "tolist"):
            value = value.tolist()
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
        return []

    df["tokens"] = df["tokens"].map(normalize_tokens)

    # Giữ thêm text_clean để script GSDMM khử văn bản trùng.
    if "text_clean" not in df.columns:
        df["text_clean"] = df["text"].str.lower().str.replace(
            r"\s+", " ", regex=True
        ).str.strip()

    # Chuẩn hóa tên cột tùy chọn mà repo GSDMM hỗ trợ.
    rename = {
        "author_channel_id": "author",
        "like_count": "likes",
        "published_at": "published",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    keep = [
        c for c in [
            "comment_id", "video_id", "author", "text", "text_clean",
            "tokens", "likes", "published", "is_reply",
        ]
        if c in df.columns
    ]
    df = df[keep].sort_values("comment_id").reset_index(drop=True)

    if df.empty:
        raise ValueError("Silver không còn bình luận hợp lệ sau khi lọc.")
    if df["comment_id"].duplicated().any():
        raise ValueError("Silver vẫn có comment_id trùng.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tạo file mới trước, sau đó thay thế snapshot cũ trong một thao tác.
    # Nếu ghi lỗi, snapshot Silver hiện tại vẫn còn nguyên.
    with tempfile.NamedTemporaryFile(
        prefix=".comments_clean-", suffix=".parquet.tmp", dir=output_dir, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        df.to_parquet(temp_path, index=False, compression="zstd")
        os.replace(temp_path, output_file)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Silver đã cập nhật: {output_file} ({len(df):,} dòng)")
    return output_file


def run_gsdmm_steps() -> Path:
    """
    Tạo kết quả trong thư mục tạm của một lần chạy. Chỉ sau khi cả GSDMM,
    Aspect và Sentiment thành công mới promote sang các đường dẫn Gold cố định.
    """
    import shutil
    from datetime import datetime, timezone

    compose_dir = os.getenv("YTB_GSDMM_COMPOSE_DIR")
    if not compose_dir:
        raise RuntimeError(
            "Chưa cấu hình YTB_GSDMM_COMPOSE_DIR. "
            "Không thể chạy GSDMM tự động từ Airflow."
        )

    compose_path = Path(compose_dir)
    if not (compose_path / "compose.yaml").is_file():
        raise FileNotFoundError(f"Không tìm thấy compose.yaml trong {compose_path}")

    # config.DATA_DIR và ../data trong compose.yaml phải trỏ tới cùng volume dữ liệu.
    # Nếu không, bước xác minh bên dưới sẽ fail closed thay vì publish nhầm output.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work_root = config.DATA_DIR / "gold" / "_building" / run_id
    gsdmm_work = work_root / "gsdmm"
    aspects_work = work_root / "aspects"
    sentiment_work = work_root / "sentiment"

    def container_path(path: Path) -> str:
        # Compose mount ../data thành /opt/data.
        relative = path.relative_to(config.DATA_DIR)
        return str(Path("/opt/data") / relative)

    def run_compose(*args):
        command = ["docker", "compose", *args]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=compose_path, check=True, timeout=6 * 60 * 60)

    work_root.mkdir(parents=True, exist_ok=False)
    try:
        run_compose(
            "run", "--rm", "gsdmm",
            "--input", "/opt/data/silver/comments_clean/comments_clean.parquet",
            "--output-dir", container_path(gsdmm_work),
            "--iterations", os.getenv("YTB_GSDMM_ITERATIONS", "30"),
        )
        clustered = gsdmm_work / "comments_clustered.parquet"
        top_words = gsdmm_work / "cluster_top_words.csv"
        if not clustered.is_file() or not top_words.is_file():
            raise FileNotFoundError(
                "Không thấy output GSDMM trên volume của Airflow. "
                "Kiểm tra mapping data/ giữa Airflow và gsdmm_sentiment/compose.yaml."
            )

        run_compose(
            "run", "--rm", "aspects",
            "--input", container_path(clustered),
            "--top-words", container_path(top_words),
            "--output-dir", container_path(aspects_work),
            "--mapping", "/opt/config/cluster_aspects_seed42.json",
        )
        aspect_file = aspects_work / "comments_with_aspects.parquet"
        if not aspect_file.is_file():
            raise FileNotFoundError(f"Không thấy output Aspect: {aspect_file}")

        run_compose(
            "run", "--rm", "sentiment",
            "--input", container_path(aspect_file),
            "--output-dir", container_path(sentiment_work),
        )
        sentiment_file = sentiment_work / "comments_sentiment.parquet"
        if not sentiment_file.is_file():
            raise FileNotFoundError(f"Không thấy output Sentiment: {sentiment_file}")

        # Validate schema trước khi promote snapshot hiện hành.
        import pyarrow.parquet as pq
        table = pq.read_table(sentiment_file)
        required = {"comment_id", "video_id", "cluster", "aspect", "sentiment"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"Sentiment thiếu cột: {sorted(missing)}")
        if table.num_rows == 0:
            raise ValueError("Sentiment output rỗng; không thay thế bản hiện hành.")
        ids = table["comment_id"].to_pylist()
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Sentiment có comment_id rỗng hoặc trùng.")

        # Các thư mục làm việc nằm cùng filesystem với các output cố định.
        # Giữ bản cũ đến khi thư mục mới đã sẵn sàng; rollback nếu promote lỗi.
        promotions = [
            (gsdmm_work, config.DATA_DIR / "gold" / "gsdmm_current"),
            (aspects_work, config.DATA_DIR / "gold" / "aspects_seed42"),
            (sentiment_work, config.DATA_DIR / "gold" / "sentiment_seed42"),
        ]
        backups = []
        promoted = []
        try:
            for _, target in promotions:
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = work_root / (target.name + ".previous")
                if target.exists():
                    target.rename(backup)
                    backups.append((backup, target))
            for source, target in promotions:
                source.rename(target)
                promoted.append(target)
        except Exception:
            for target in reversed(promoted):
                if target.exists():
                    shutil.rmtree(target)
            for backup, target in reversed(backups):
                if backup.exists():
                    backup.rename(target)
            raise

        final_file = config.DATA_DIR / "gold" / "sentiment_seed42" / "comments_sentiment.parquet"
        print(f"Đã cập nhật Gold hiện hành: {final_file} ({table.num_rows:,} dòng)", flush=True)
        return final_file
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def run_business() -> dict:
    silver_file = build_silver()

    # Nếu bạn đã chạy GSDMM thủ công, có thể đặt YTB_SKIP_GSDMM=1
    # và sử dụng kết quả có sẵn; không chạy lại/ghi đè model.
    if os.getenv("YTB_SKIP_GSDMM", "0") == "1":
        result_file = (
            config.DATA_DIR / "gold" / "sentiment_seed42"
            / "comments_sentiment.parquet"
        )
        if not result_file.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy kết quả sentiment có sẵn: {result_file}"
            )
    else:
        result_file = run_gsdmm_steps()
        if not result_file.is_file():
            raise FileNotFoundError(
                f"GSDMM chạy xong nhưng thiếu output: {result_file}"
            )

    import pyarrow.parquet as pq

    result = pq.read_table(result_file)
    required = {"comment_id", "video_id", "cluster", "aspect", "sentiment"}
    missing = required - set(result.column_names)
    if missing:
        raise ValueError(f"Kết quả GSDMM thiếu cột: {sorted(missing)}")

    business_dir = config.DATA_DIR / "business"
    business_dir.mkdir(parents=True, exist_ok=True)
    business_file = business_dir / "fact_comment_sentiment.parquet"

    # Snapshot Business phản ánh đúng lần xử lý Silver hiện tại. Ghi file tạm
    # rồi thay thế để retry không lỗi vì file cũ, đồng thời tránh Parquet dở dang.
    with tempfile.NamedTemporaryFile(
        prefix=".fact_comment_sentiment-", suffix=".parquet.tmp",
        dir=business_dir, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        pq.write_table(result, temp_path, compression="zstd")
        os.replace(temp_path, business_file)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "silver_file": str(silver_file),
        "business_file": str(business_file),
        "business_rows": result.num_rows,
    }


if __name__ == "__main__":
    print(run_business())