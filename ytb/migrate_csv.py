"""Chạy MỘT LẦN: nạp dữ liệu CSV cũ (từ ytb_data.py) vào Staging dạng Parquet.

Chuẩn bị: chép 3 file vào thư mục data/legacy/ trên máy:
    videos_nhac_ai.csv, comments_nhac_ai.csv, backfill_state.json   (backfill_state.json nếu có)
Chạy:
    docker compose --profile debug run --rm airflow-cli python -m ytb.migrate_csv
Chạy lại nhiều lần cũng an toàn: dòng nào đã có thì bỏ qua.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .storage import comment_index, iso, known_video_ids, load_state, save_state, write_rows

log = logging.getLogger(__name__)
CHUNK = 50000


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_time(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def migrate_videos(path: Path) -> int:
    when = _file_time(path)
    known = known_video_ids()
    rows = []
    for r in _read_csv(path):
        if not r.get("video_id") or r["video_id"] in known:
            continue
        known.add(r["video_id"])
        rows.append({
            "video_id": r["video_id"], "keyword": r.get("keyword") or None, "title": r.get("title"),
            "channel_title": r.get("channel"), "published_at": r.get("published") or None,
            "source": "legacy_csv", "fetched_at": iso(when),
        })
    write_rows("videos", rows, when.strftime("%Y-%m-%d"))
    return len(rows)


def migrate_comments(path: Path) -> int:
    when = _file_time(path)
    updated, _ = comment_index()
    rows, total = [], 0
    for r in _read_csv(path):
        cid = r.get("comment_id")
        if not cid or cid in updated:
            continue
        updated[cid] = None
        rows.append({
            "comment_id": cid, "video_id": r.get("video_id"), "thread_id": cid, "parent_id": None,
            "is_reply": False, "author_display_name": r.get("author"), "text": r.get("text"),
            "like_count": _int(r.get("likes")), "published_at": r.get("published") or None,
            "updated_at": None, "source": "legacy_csv", "fetched_at": iso(when),
        })
        if len(rows) >= CHUNK:
            write_rows("comments", rows, when.strftime("%Y-%m-%d"))
            total += len(rows)
            rows = []
    write_rows("comments", rows, when.strftime("%Y-%m-%d"))
    return total + len(rows)


def migrate_backfill_state(path: Path) -> bool:
    """Giữ nguyên tiến độ backfill cũ để khỏi tìm lại từ đầu."""
    if load_state("backfill_state.json", None) is not None:
        return False
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if "next_before" not in state:
        return False
    save_state("backfill_state.json", {"next_before": state["next_before"]})
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=config.DATA_DIR / "legacy", help="thư mục chứa các file CSV cũ")
    src = ap.parse_args().src

    videos, comments, state = src / "videos_nhac_ai.csv", src / "comments_nhac_ai.csv", src / "backfill_state.json"
    if not videos.exists() and not comments.exists():
        raise SystemExit(f"Không thấy file CSV nào trong {src}. Hãy chép các file cũ vào data/legacy/ trước.")
    if videos.exists():
        log.info("Video đã nạp: %d", migrate_videos(videos))
    if comments.exists():
        log.info("Bình luận đã nạp: %d", migrate_comments(comments))
    if state.exists():
        log.info("Tiến độ backfill: %s", "đã chép" if migrate_backfill_state(state) else "giữ nguyên (đã có)")
    log.info("Xong. Dùng `python -m ytb.status` để xem số liệu.")


if __name__ == "__main__":
    main()