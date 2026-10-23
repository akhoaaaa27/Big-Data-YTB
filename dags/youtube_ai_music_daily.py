"""
DAG chính: thu thập và xử lý phản ứng khán giả Việt Nam với nhạc AI trên YouTube.

BƯỚC 2: ba task thu thập đã chạy thật (ghi Parquet vào data/staging/).
Các task Spark và publish vẫn là khung, sẽ làm ở bước 3 và 4.

  discover_videos -> snapshot_stats -> fetch_comments
      -> spark_core -> spark_business -> spark_mart -> publish
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import DAG, task

TZ = "Asia/Ho_Chi_Minh"


def _placeholder(ten_task: str, se_lam: str) -> None:
    print(f"[{ten_task}] Chưa làm ở bước này. Task này sẽ: {se_lam}")


with DAG(
    dag_id="youtube_ai_music_daily",
    description="Thu thập, xử lý và đưa phản ứng khán giả về nhạc AI lên BigQuery mỗi ngày",
    # Chạy lúc 20:00 giờ Việt Nam mỗi ngày. Đổi giờ này cho khớp lúc máy bạn đang bật.
    schedule="0 20 * * *",
    start_date=pendulum.datetime(2026, 10, 19, tz=TZ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "khoa",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["ytb", "nhac-ai"],
    doc_md=__doc__,
) as dag:

    @task 
    def validate_ml_runtime(): 
        # Fail early if Docker Desktop/socket or dedicated ML runner containers are unavailable. 
        from ytb.ml_containers import check_ml_runtime, run_ml_job 
 
        result = check_ml_runtime() 
        run_ml_job( 
            "ytb-gsdmm-runner", 
            [ 
                "gsdmm_train.py", 
                "--mode", "check", 
                "--model-dir", "/opt/data/models/gsdmm_v1", 
            ], 
        ) 
        return result

    @task
    def discover_videos():
        from ytb.discover import run_discover   # import trong task để lỗi thư viện không làm hỏng việc đọc DAG

        return run_discover()

    @task
    def snapshot_stats():
        from ytb.snapshot import run_snapshot

        return run_snapshot()

    @task(execution_timeout=timedelta(hours=4))
    def fetch_comments():
        from ytb.comments import run_comments

        return run_comments()

    @task(execution_timeout=timedelta(hours=2))
    def spark_core():
        from ytb.spark_core import run_core

        return run_core()

    @task
    def spark_business():
        #_placeholder("spark_business", "gán khía cạnh (GSDMM) và cảm xúc, ra tầng Business")
        from ytb.spark_business import run_business

        return run_business() 
    
    @task
    def spark_mart():
        #_placeholder("spark_mart", "tổng hợp theo video/thời gian/khía cạnh, ghi PostgreSQL (ytb_mart)")
        from ytb.spark_mart import run_mart

        return run_mart()
    
    (
        validate_ml_runtime()
        >> discover_videos()
        >> snapshot_stats()
        >> fetch_comments()
        >> spark_core()
        >> spark_business()
        >> spark_mart()
    )