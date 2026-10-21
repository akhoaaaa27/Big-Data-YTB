"""Lưu dữ liệu Staging: Parquet ghi nối thêm (không sửa, không xóa), chia theo ingest_date.

data/staging/<bảng>/ingest_date=YYYY-MM-DD/part-<giờ>-<mã>.parquet
Mỗi lần ghi tạo một file mới. Việc khử trùng (giữ bản mới nhất theo comment_id) làm ở tầng Core.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import config

SCHEMAS = {
    "videos": pa.schema([
        ("video_id", pa.string()),
        ("keyword", pa.string()),
        ("title", pa.string()),
        ("channel_id", pa.string()),
        ("channel_title", pa.string()),
        ("description", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("published_at", pa.string()),
        ("category_id", pa.string()),
        ("default_language", pa.string()),
        ("default_audio_language", pa.string()),
        ("duration", pa.string()),
        ("contains_synthetic_media", pa.bool_()),
        ("source", pa.string()),        # api | legacy_csv
        ("fetched_at", pa.string()),
    ]),
    "video_snapshots": pa.schema([
        ("video_id", pa.string()),
        ("snapshot_date", pa.string()),
        ("snapshot_at", pa.string()),
        ("view_count", pa.int64()),
        ("like_count", pa.int64()),
        ("comment_count", pa.int64()),
    ]),
    "comments": pa.schema([
        ("comment_id", pa.string()),
        ("video_id", pa.string()),
        ("thread_id", pa.string()),      # id của bình luận gốc (với bình luận gốc thì trùng comment_id)
        ("parent_id", pa.string()),      # rỗng nếu là bình luận gốc
        ("is_reply", pa.bool_()),
        ("author_display_name", pa.string()),
        ("author_channel_id", pa.string()),
        ("text", pa.string()),
        ("like_count", pa.int64()),
        ("published_at", pa.string()),
        ("updated_at", pa.string()),
        ("total_reply_count", pa.int64()),   # chỉ có ở bình luận gốc
        ("source", pa.string()),             # api | legacy_csv
        ("fetched_at", pa.string()),
    ]),
}


# ---------- thời gian ----------
def iso(dt: datetime) -> str:
    return dt.strftime(config.TIME_FMT)


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, config.TIME_FMT).replace(tzinfo=timezone.utc)


def today_str(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


# ---------- Parquet ----------
def table_dir(table: str) -> Path:
    return config.STAGING_DIR / table


def write_rows(table: str, rows: list[dict], ingest_date: str | None = None) -> Path | None:
    """Ghi một file Parquet mới vào đúng partition. Không có dòng nào thì không ghi gì."""
    if not rows:
        return None
    ingest_date = ingest_date or today_str()
    folder = table_dir(table) / f"ingest_date={ingest_date}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"part-{datetime.now(timezone.utc):%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
    path = folder / name
    tmp = folder / (name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[table]), tmp, compression="zstd")
    os.replace(tmp, path)   # ghi xong mới đổi tên, tránh file dở dang
    return path


def read_columns(table: str, columns: list[str]) -> dict[str, list]:
    """Đọc vài cột của toàn bộ bảng (mọi partition). Bảng chưa có dữ liệu thì trả về cột rỗng."""
    folder = table_dir(table)
    if not folder.exists() or not any(folder.rglob("*.parquet")):
        return {c: [] for c in columns}
    dataset = ds.dataset(str(folder), format="parquet", partitioning="hive")
    return dataset.to_table(columns=columns).to_pydict()


def count_rows(table: str) -> int:
    folder = table_dir(table)
    if not folder.exists() or not any(folder.rglob("*.parquet")):
        return 0
    return ds.dataset(str(folder), format="parquet", partitioning="hive").count_rows()


# ---------- những gì đã biết (để khỏi ghi trùng) ----------
def known_video_ids() -> set[str]:
    return set(read_columns("videos", ["video_id"])["video_id"])


def video_meta() -> dict[str, dict]:
    """Mỗi video: published_at, keyword, và có đủ siêu dữ liệu mới (channel_id) hay chưa."""
    cols = read_columns("videos", ["video_id", "keyword", "published_at", "channel_id"])
    meta: dict[str, dict] = {}
    for vid, kw, pub, ch in zip(cols["video_id"], cols["keyword"], cols["published_at"], cols["channel_id"]):
        m = meta.setdefault(vid, {"keyword": None, "published_at": None, "complete": False})
        m["keyword"] = m["keyword"] or kw
        m["published_at"] = m["published_at"] or pub
        m["complete"] = m["complete"] or ch is not None
    return meta


def comment_index() -> tuple[dict[str, str | None], dict[str, int]]:
    """(comment_id -> updated_at đã lưu, thread_id -> số reply đã lưu)."""
    cols = read_columns("comments", ["comment_id", "thread_id", "is_reply", "updated_at"])
    updated: dict[str, str | None] = {}
    replies: dict[str, int] = {}
    seen_reply: set[str] = set()
    for cid, tid, is_reply, upd in zip(cols["comment_id"], cols["thread_id"], cols["is_reply"], cols["updated_at"]):
        updated[cid] = upd if upd is not None else updated.get(cid)
        if is_reply and cid not in seen_reply:
            seen_reply.add(cid)
            replies[tid] = replies.get(tid, 0) + 1
    return updated, replies


def latest_snapshot_counts() -> dict[str, int | None]:
    """comment_count trong snapshot mới nhất của từng video."""
    cols = read_columns("video_snapshots", ["video_id", "snapshot_at", "comment_count"])
    best: dict[str, tuple[str, int | None]] = {}
    for vid, at, cnt in zip(cols["video_id"], cols["snapshot_at"], cols["comment_count"]):
        if vid not in best or at > best[vid][0]:
            best[vid] = (at, cnt)
    return {vid: cnt for vid, (_, cnt) in best.items()}


def snapshot_video_ids_on(date_str: str) -> set[str]:
    cols = read_columns("video_snapshots", ["video_id", "snapshot_date"])
    return {v for v, d in zip(cols["video_id"], cols["snapshot_date"]) if d == date_str}


# ---------- file trạng thái JSON ----------
def load_state(name: str, default):
    path = config.STATE_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_state(name: str, data) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.STATE_DIR / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
