
"""Publish PostgreSQL Mart tables to BigQuery."""

from __future__ import annotations

import os


TABLES = {
    "fact_comment_sentiment": "fact_comment_sentiment",
    "agg_aspect_sentiment": "agg_aspect_sentiment",
    "agg_video_sentiment": "agg_video_sentiment",
}


def run_publish() -> dict:
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID")

    # Pipeline được yêu cầu chạy end-to-end. Không báo task xanh khi publish
    # thực tế chưa chạy; cấu hình BigQuery trước khi bật DAG production.
    missing_bq = [
        name for name, value in {
            "BIGQUERY_PROJECT_ID": project_id,
            "BIGQUERY_DATASET_ID": dataset_id,
        }.items() if not value
    ]
    if missing_bq:
        raise RuntimeError(
            "Chưa thể publish lên BigQuery; thiếu cấu hình: "
            + ", ".join(missing_bq)
            + ". Hãy cấu hình biến môi trường và credentials của Google Cloud."
        )

    from google.cloud import bigquery
    import pandas as pd

    import psycopg2

    required = ["MART_PGHOST", "MART_PGUSER", "MART_PGPASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Thiếu cấu hình PostgreSQL: {missing}")

    pg_conn = psycopg2.connect(
        host=os.environ["MART_PGHOST"],
        port=int(os.getenv("MART_PGPORT", "5432")),
        dbname=os.getenv("MART_PGDATABASE", "ytb_mart"),
        user=os.environ["MART_PGUSER"],
        password=os.environ["MART_PGPASSWORD"],
        connect_timeout=10,
    )

    bq_client = bigquery.Client(project=project_id)
    results = {}

    try:
        for pg_table, bq_table in TABLES.items():
            df = pd.read_sql_query(
                f'SELECT * FROM "{pg_table}"',
                pg_conn,
            )

            table_id = f"{project_id}.{dataset_id}.{bq_table}"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                autodetect=True,
            )

            # Ghi lại toàn bộ snapshot Mart của bảng này.
            job = bq_client.load_table_from_dataframe(
                df,
                table_id,
                job_config=job_config,
            )
            job.result()  # Chờ BigQuery xác nhận job hoàn tất.

            table = bq_client.get_table(table_id)
            results[bq_table] = {
                "rows": table.num_rows,
                "table": table_id,
                "job_id": job.job_id,
            }

    finally:
        pg_conn.close()

    return {
        "status": "success",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "tables": results,
    }


if __name__ == "__main__":
    print(run_publish())