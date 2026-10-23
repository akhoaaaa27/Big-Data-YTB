
"""Business layer: Core -> Silver -> GSDMM/aspect/sentiment -> Business."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from . import config
from .ml_containers import run_ml_job


def build_silver() -> Path: 
    """Dùng chung bộ lọc với clean_comments.py; chỉ thay Silver sau khi ghi xong.""" 
    from gsdmm_sentiment.jobs.clean_comments import run_clean 
    from .spark_core import build_spark 
 
    source = config.DATA_DIR / "core" / "comments" 
    target = config.DATA_DIR / "silver" / "comments_clean" 
    audit = config.DATA_DIR / "gold" / "comments_removed_offtopic" 
    if not source.is_dir() or not any(source.glob("*.parquet")): 
        raise FileNotFoundError(f"Không thấy Core Parquet: {source}") 
    target.parent.mkdir(parents=True, exist_ok=True) 
    with tempfile.TemporaryDirectory(prefix=".silver-build-", dir=target.parent) as work: 
        candidate = Path(work) / "new" 
        previous = Path(work) / "previous" 
        spark = build_spark("ytb_silver") 
        try: 
            run_clean(source, candidate, audit, spark=spark) 
        finally: 
            spark.stop() 
        if not any(candidate.glob("*.parquet")): 
            raise RuntimeError("Không có Silver Parquet mới.") 
        if target.exists(): 
            target.rename(previous) 
        try: 
            candidate.rename(target) 
        except Exception: 
            if previous.exists(): 
                previous.rename(target) 
            raise 
    print(f"Silver đã cập nhật: {target}", flush=True) 
    return target 
 
 
def run_gsdmm_steps() -> Path: 
    """Dùng model cố định để gán cụm/khía cạnh rồi chạy sentiment.""" 
    import os 
    from datetime import datetime, timezone 
    import pyarrow.parquet as pq 
 
    model_dir = config.DATA_DIR / "models" / "gsdmm_v1" 
    silver_dir = config.DATA_DIR / "silver" / "comments_clean" 
 
    if not (model_dir / "model_manifest.json").is_file(): 
        raise FileNotFoundError("Cần chuẩn bị và khóa model gsdmm_v1 trước.") 
    if not silver_dir.is_dir() or not any(silver_dir.glob("*.parquet")): 
        raise FileNotFoundError(f"Không thấy Silver: {silver_dir}") 
 
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") 
    relative = f"gold/daily/{run_id}" 
    local_run = config.DATA_DIR / relative 
    local_run.mkdir(parents=True, exist_ok=False) 
 
    run_ml_job( 
        "ytb-gsdmm-runner", 
        [ 
            "gsdmm_train.py", 
            "--mode", "predict", 
            "--model-dir", "/opt/data/models/gsdmm_v1", 
            "--input", "/opt/data/silver/comments_clean", 
            "--output-dir", f"/opt/data/{relative}/aspects", 
        ], 
    ) 
 
    run_ml_job( 
        "ytb-sentiment-runner", 
        [ 
            "predict_sentiment.py", 
            "--input", f"/opt/data/{relative}/aspects/comments_with_aspects.parquet", 
            "--output-dir", f"/opt/data/{relative}/sentiment", 
            "--batch-size", os.getenv("YTB_SENTIMENT_BATCH_SIZE", "16"), 
            "--cpu-threads", os.getenv("YTB_SENTIMENT_CPU_THREADS", "4"), 
            "--device", "cpu", 
        ], 
    ) 
 
    result_file = local_run / "sentiment" / "comments_sentiment.parquet" 
    table = pq.read_table(result_file) 
    required = {"comment_id", "video_id", "cluster", "aspect", "sentiment"} 
    if not required <= set(table.column_names) or table.num_rows == 0: 
        raise ValueError("Kết quả sentiment rỗng hoặc thiếu cột.") 
    ids = table["comment_id"].to_pylist() 
    if any(not value for value in ids) or len(ids) != len(set(ids)): 
        raise ValueError("Kết quả có ID rỗng hoặc trùng.") 
    print(f"Đã xử lý bằng model cố định: {result_file}", flush=True) 
    return result_file 
 
 
def run_business() -> dict: 
    silver_dir = build_silver() 
    result_file = run_gsdmm_steps() 
    if not result_file.is_file(): 
        raise FileNotFoundError(f"Thiếu kết quả mô hình: {result_file}") 
 
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
        "silver_dir": str(silver_dir), 
        "business_file": str(business_file), 
        "business_rows": result.num_rows, 
    }


if __name__ == "__main__":
    print(run_business())