"""
DAG kiểm tra hạ tầng, chạy bằng tay một lần để chắc chắn mọi thứ nối đúng:
  1. Thư mục data/ trên máy bạn được gắn vào container (ghi file thử).
  2. Kết nối được PostgreSQL và có database ytb_mart.
"""

from __future__ import annotations

import os
from pathlib import Path

import pendulum
from airflow.sdk import DAG, task

TZ = "Asia/Ho_Chi_Minh"

with DAG(
    dag_id="smoke_test",
    description="Kiểm tra thư mục dữ liệu và kết nối PostgreSQL",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 19, tz=TZ),
    catchup=False,
    tags=["ytb", "kiem-tra"],
    doc_md=__doc__,
) as dag:

    @task
    def kiem_tra_thu_muc_du_lieu():
        file_thu = Path("/opt/airflow/data/staging/_smoke_test.txt")
        file_thu.parent.mkdir(parents=True, exist_ok=True)
        file_thu.write_text(f"ok {pendulum.now(TZ).isoformat()}\n", encoding="utf-8")
        print(f"Đã ghi {file_thu}. Trên máy bạn, file nằm ở data/staging/_smoke_test.txt")

    @task
    def kiem_tra_postgres():
        import psycopg2

        conn = psycopg2.connect(
            host="postgres",
            port=5432,
            dbname="ytb_mart",
            user=os.environ.get("POSTGRES_USER", "airflow"),
            password=os.environ["POSTGRES_PASSWORD"],
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version(), current_database()")
                phien_ban, ten_db = cur.fetchone()
            print(f"Kết nối OK. Database: {ten_db}")
            print(f"PostgreSQL: {phien_ban}")
        finally:
            conn.close()

    kiem_tra_thu_muc_du_lieu()
    kiem_tra_postgres()
