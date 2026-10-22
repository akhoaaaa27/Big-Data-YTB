
"""Load Business Parquet into PostgreSQL ytb_mart."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config


def _connect():
    import psycopg2

    # Các biến này cần được cấu hình trong environment của Airflow.
    required = ["MART_PGHOST", "MART_PGUSER", "MART_PGPASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Thiếu cấu hình PostgreSQL: {missing}")

    return psycopg2.connect(
        host=os.environ["MART_PGHOST"],
        port=int(os.getenv("MART_PGPORT", "5432")),
        dbname=os.getenv("MART_PGDATABASE", "ytb_mart"),
        user=os.environ["MART_PGUSER"],
        password=os.environ["MART_PGPASSWORD"],
        connect_timeout=10,
    )


def run_mart() -> dict:
    import psycopg2.extras

    source = (
        config.DATA_DIR / "business" / "fact_comment_sentiment.parquet"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy Business Parquet: {source}. "
            "Hãy chạy spark_business trước."
        )

    df = pd.read_parquet(source)
    required = {
        "comment_id", "video_id", "cluster", "aspect", "sentiment"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Business thiếu cột: {sorted(missing)}")

    df = df.dropna(subset=["comment_id", "video_id"]).copy()
    df["comment_id"] = df["comment_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["aspect"] = df["aspect"].fillna("Khác").astype(str)
    df["sentiment"] = df["sentiment"].fillna("Chưa dự đoán").astype(str)

    if df["comment_id"].duplicated().any():
        raise ValueError("Business có comment_id trùng; dừng nạp Mart.")

    # Chỉ nạp các cột đã xác định schema, tránh tự tạo cột SQL không kiểm soát.
    allowed = [
        "comment_id", "video_id", "cluster", "aspect", "sentiment",
        "sentiment_score", "sentiment_negative", "sentiment_neutral",
        "sentiment_positive", "published", "likes", "text",
    ]
    columns = [c for c in allowed if c in df.columns]
    records = df[columns].where(pd.notna(df[columns]), None).to_dict("records")

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS fact_comment_sentiment (
                        comment_id TEXT PRIMARY KEY,
                        video_id TEXT NOT NULL,
                        cluster INTEGER,
                        aspect TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        sentiment_score DOUBLE PRECISION,
                        sentiment_negative DOUBLE PRECISION,
                        sentiment_neutral DOUBLE PRECISION,
                        sentiment_positive DOUBLE PRECISION,
                        published TIMESTAMPTZ,
                        likes BIGINT,
                        text TEXT,
                        loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)

                # Nạp theo lô và cập nhật những comment đã tồn tại.
                sql = """
                    INSERT INTO fact_comment_sentiment (
                        comment_id, video_id, cluster, aspect, sentiment,
                        sentiment_score, sentiment_negative,
                        sentiment_neutral, sentiment_positive,
                        published, likes, text, loaded_at
                    )
                    VALUES (
                        %(comment_id)s, %(video_id)s, %(cluster)s,
                        %(aspect)s, %(sentiment)s, %(sentiment_score)s,
                        %(sentiment_negative)s, %(sentiment_neutral)s,
                        %(sentiment_positive)s, %(published)s,
                        %(likes)s, %(text)s, NOW()
                    )
                    ON CONFLICT (comment_id) DO UPDATE SET
                        video_id = EXCLUDED.video_id,
                        cluster = EXCLUDED.cluster,
                        aspect = EXCLUDED.aspect,
                        sentiment = EXCLUDED.sentiment,
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_negative = EXCLUDED.sentiment_negative,
                        sentiment_neutral = EXCLUDED.sentiment_neutral,
                        sentiment_positive = EXCLUDED.sentiment_positive,
                        published = EXCLUDED.published,
                        likes = EXCLUDED.likes,
                        text = EXCLUDED.text,
                        loaded_at = NOW()
                """

                if records:
                    psycopg2.extras.execute_batch(
                        cur, sql, records, page_size=1000
                    )

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agg_aspect_sentiment (
                        aspect TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        comment_count BIGINT NOT NULL,
                        total_for_aspect BIGINT NOT NULL,
                        percent_within_aspect DOUBLE PRECISION NOT NULL,
                        refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (aspect, sentiment)
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agg_video_sentiment (
                        video_id TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        comment_count BIGINT NOT NULL,
                        total_for_video BIGINT NOT NULL,
                        percent_within_video DOUBLE PRECISION NOT NULL,
                        refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (video_id, sentiment)
                    )
                """)

                # Xóa và dựng lại bảng tổng hợp bên trong cùng transaction.
                # Nếu lỗi trước khi commit, PostgreSQL rollback cả phần xóa.
                cur.execute("DELETE FROM agg_aspect_sentiment")
                cur.execute("""
                    INSERT INTO agg_aspect_sentiment (
                        aspect, sentiment, comment_count, total_for_aspect,
                        percent_within_aspect, refreshed_at
                    )
                    WITH counts AS (
                        SELECT aspect, sentiment, COUNT(*) AS n
                        FROM fact_comment_sentiment
                        GROUP BY aspect, sentiment
                    )
                    SELECT aspect, sentiment, n,
                           SUM(n) OVER (PARTITION BY aspect),
                           ROUND(
                               100.0 * n /
                               NULLIF(SUM(n) OVER (PARTITION BY aspect), 0),
                               2
                           ),
                           NOW()
                    FROM counts
                """)

                cur.execute("DELETE FROM agg_video_sentiment")
                cur.execute("""
                    INSERT INTO agg_video_sentiment (
                        video_id, sentiment, comment_count, total_for_video,
                        percent_within_video, refreshed_at
                    )
                    WITH counts AS (
                        SELECT video_id, sentiment, COUNT(*) AS n
                        FROM fact_comment_sentiment
                        GROUP BY video_id, sentiment
                    )
                    SELECT video_id, sentiment, n,
                           SUM(n) OVER (PARTITION BY video_id),
                           ROUND(
                               100.0 * n /
                               NULLIF(SUM(n) OVER (PARTITION BY video_id), 0),
                               2
                           ),
                           NOW()
                    FROM counts
                """)

                cur.execute("SELECT COUNT(*) FROM fact_comment_sentiment")
                fact_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM agg_aspect_sentiment")
                aspect_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM agg_video_sentiment")
                video_count = cur.fetchone()[0]

        return {
            "database": os.getenv("MART_PGDATABASE", "ytb_mart"),
            "fact_rows": fact_count,
            "aspect_aggregate_rows": aspect_count,
            "video_aggregate_rows": video_count,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(run_mart())
