"""Cấu hình chung. Các giá trị lấy nguyên từ ytb_data.py cũ, chỉ thêm phần mới."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# ---- Thư mục dữ liệu (trong container là /opt/airflow/data, ứng với thư mục data/ trên máy) ----
DATA_DIR = Path(os.getenv("YTB_DATA_DIR", "/opt/airflow/data"))
STAGING_DIR = DATA_DIR / "staging"   # Parquet, ghi nối thêm, chia theo ingest_date
STATE_DIR = DATA_DIR / "state"       # tiến độ backfill, trạng thái quét bình luận

TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

# ---- Tìm video (giữ nguyên như script cũ) ----
KEYWORDS = [
    '"nhạc AI"', '"bài hát AI"', '"nhạc do AI sáng tác"', '"bài hát do AI tạo"',
    '"nhạc AI Việt Nam"', "Suno nhạc Việt", "Suno AI bài hát", "Udio bài hát",
    '"AI cover"', '"cover bằng AI"', '"giọng hát AI"', '"ca sĩ AI"',
    '"AI trong âm nhạc"', "tranh cãi nhạc AI", '"nhạc AI" có hay không', "nhạc AI thay thế ca sĩ",
]
VIDEOS_PER_KEYWORD = 50
REGION_CODE = "VN"
RELEVANCE_LANGUAGE = "vi"
DAYS_BACK = 3                 # phần video mới (mỗi ngày)
WINDOW_DAYS = 30              # video nhạc AI thưa nên mỗi cửa sổ lùi một tháng
WINDOWS_PER_RUN = 1          # số cửa sổ quá khứ mỗi lần chạy
BACKFILL_STOP = datetime(2024, 1, 1, tzinfo=timezone.utc)

# ---- Quét lại bình luận theo độ tuổi video (đề xuất của nhóm thảo luận, chỉnh được) ----
RESCAN_DAILY_MAX_AGE_DAYS = 30      # video <= 30 ngày tuổi: quét lại tối đa mỗi ngày
RESCAN_WEEKLY_MAX_AGE_DAYS = 90     # 30-90 ngày tuổi: quét lại tối đa mỗi tuần
RESCAN_WEEKLY_INTERVAL_DAYS = 7     # video > 90 ngày tuổi: chỉ quét đầy đủ một lần

# Mỗi lệnh search thật ra tốn 100 units trong chung 10.000 units/ngày (bạn nên tự kiểm tra lại trên Google Cloud).
# 34 search x 100 = 3.400, + comments 5.100 + khác 500 = 9.000.
QUOTA_BUDGET_COMMENTS = int(os.getenv("YTB_QUOTA_BUDGET_COMMENTS", "5100"))
QUOTA_BUDGET_OTHER = int(os.getenv("YTB_QUOTA_BUDGET_OTHER", "500"))
SEARCH_CALL_BUDGET = int(os.getenv("YTB_SEARCH_CALL_BUDGET", "34"))
