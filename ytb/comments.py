"""Task fetch_comments: lấy toàn bộ bình luận gồm cả reply, quét lại theo độ tuổi video."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import config
from .storage import (
    comment_index, iso, latest_snapshot_counts, load_state, parse_iso,
    save_state, today_str, video_meta, write_rows,
)
from .youtube_client import (
    Budget, BudgetExceeded, CommentsUnavailable, QuotaExhausted, build_client, call,
)

log = logging.getLogger(__name__)
STATE_FILE = "comment_scans.json"   # {video_id: {status, last_scan_at, scan_comment_count}}
FLUSH_EVERY_ROWS = 20000            # gom đủ ngần này dòng mới ghi một file Parquet
SAVE_STATE_EVERY = 50               # lưu trạng thái sau mỗi 50 video


def comment_row(c: dict, video_id: str, thread_id: str, parent_id: str | None,
                total_reply_count: int | None, now: datetime) -> dict:
    sn = c["snippet"]
    published = sn.get("publishedAt")
    return {
        "comment_id": c["id"],
        "video_id": sn.get("videoId") or video_id,
        "thread_id": thread_id,
        "parent_id": parent_id,
        "is_reply": parent_id is not None,
        "author_display_name": sn.get("authorDisplayName"),
        "author_channel_id": (sn.get("authorChannelId") or {}).get("value"),
        "text": sn.get("textDisplay"),
        "like_count": sn.get("likeCount", 0),
        "published_at": published,
        "updated_at": sn.get("updatedAt") or published,
        "total_reply_count": total_reply_count,
        "source": "api",
        "fetched_at": iso(now),
    }


class Collector:
    """Gom các dòng mới trong bộ nhớ, bỏ qua bình luận đã có (trừ khi nội dung được sửa)."""

    def __init__(self, updated: dict, reply_counts: dict):
        self.updated = updated              # comment_id -> updated_at đã lưu
        self.reply_counts = reply_counts    # thread_id -> số reply đã lưu
        self.buffer: list[dict] = []
        self.new_top = 0
        self.new_replies = 0

    def add(self, row: dict) -> bool:
        cid = row["comment_id"]
        known = cid in self.updated
        # Bình luận cũ từ CSV không có updated_at (None): coi như đã có, không ghi lại.
        if known and (self.updated[cid] is None or self.updated[cid] == row["updated_at"]):
            return False
        self.updated[cid] = row["updated_at"]
        self.buffer.append(row)
        if not known:
            if row["is_reply"]:
                self.reply_counts[row["thread_id"]] = self.reply_counts.get(row["thread_id"], 0) + 1
                self.new_replies += 1
            else:
                self.new_top += 1
        return True

    def flush(self, day: str, force: bool = False) -> None:
        if self.buffer and (force or len(self.buffer) >= FLUSH_EVERY_ROWS):
            write_rows("comments", self.buffer, day)
            self.buffer = []


def fetch_all_replies(client, budget: Budget, thread_id: str) -> list[dict]:
    """Lấy đủ reply của một bình luận gốc (commentThreads chỉ trả về một phần)."""
    out, token = [], None
    while True:
        res = call(client.comments().list(
            part="snippet", parentId=thread_id, maxResults=100,
            textFormat="plainText", pageToken=token), budget)
        out.extend(res.get("items", []))
        token = res.get("nextPageToken")
        if not token:
            return out


def scan_video(client, budget: Budget, video_id: str, col: Collector, now: datetime, day: str) -> None:
    token = None
    while True:
        res = call(client.commentThreads().list(
            part="snippet,replies", videoId=video_id, maxResults=100, order="time",
            textFormat="plainText", pageToken=token), budget)
        for thread in res.get("items", []):
            tid = thread["id"]
            total = thread["snippet"].get("totalReplyCount", 0)
            col.add(comment_row(thread["snippet"]["topLevelComment"], video_id, tid, None, total, now))
            replies = thread.get("replies", {}).get("comments", [])
            # Chỉ gọi thêm API khi còn thiếu reply so với tổng và chưa lưu đủ
            if total > len(replies) and total > col.reply_counts.get(tid, 0):
                replies = fetch_all_replies(client, budget, tid)
            for r in replies:
                col.add(comment_row(r, video_id, tid, r["snippet"].get("parentId") or tid, None, now))
        col.flush(day)
        token = res.get("nextPageToken")
        if not token:
            return


def select_videos(now: datetime, meta: dict, counts: dict, scans: dict) -> list[str]:
    """Thứ tự ưu tiên: video chưa quét lần nào (mới đăng trước), rồi video có thêm bình luận."""
    never, rescan = [], []
    today = now.date()
    for vid, m in meta.items():
        cc = counts.get(vid)
        if cc is None:                       # chưa có snapshot (video không còn truy cập được)
            continue
        st = scans.get(vid)
        if st is None:
            never.append((m.get("published_at") or "", vid))
            continue
        if st.get("status") == "unavailable":
            continue
        grown = cc - st.get("scan_comment_count", 0)
        if grown <= 0:
            continue
        pub = parse_iso(m["published_at"]) if m.get("published_at") else None
        age_days = (now - pub).days if pub else 10**6
        days_since = (today - parse_iso(st["last_scan_at"]).date()).days
        if ((age_days <= config.RESCAN_DAILY_MAX_AGE_DAYS and days_since >= 1)
                or (age_days <= config.RESCAN_WEEKLY_MAX_AGE_DAYS
                    and days_since >= config.RESCAN_WEEKLY_INTERVAL_DAYS)):
            rescan.append((grown, vid))
    never.sort(reverse=True)
    rescan.sort(reverse=True)
    return [v for _, v in never] + [v for _, v in rescan]


def run_comments(client=None, now: datetime | None = None, budget: Budget | None = None) -> dict:
    client = client or build_client()
    now = now or datetime.now(timezone.utc)
    budget = budget or Budget(config.QUOTA_BUDGET_COMMENTS)
    day = today_str(now)

    scans = load_state(STATE_FILE, {})
    counts = latest_snapshot_counts()
    queue = select_videos(now, video_meta(), counts, scans)
    updated, reply_counts = comment_index()
    col = Collector(updated, reply_counts)
    pending: dict[str, dict] = {}
    summary = {"queue": len(queue), "videos_scanned": 0, "unavailable": 0, "stopped": None}
    processed = 0

    def commit() -> None:
        col.flush(day, force=True)           # ghi dữ liệu trước, lưu tiến độ sau
        scans.update(pending)
        save_state(STATE_FILE, scans)
        pending.clear()

    try:
        for vid in queue:
            cc = counts[vid]
            if cc == 0:                      # không có bình luận: khỏi tốn quota
                pending[vid] = {"status": "ok", "last_scan_at": iso(now), "scan_comment_count": 0}
                processed += 1
            else:
                try:
                    scan_video(client, budget, vid, col, now, day)
                except CommentsUnavailable as e:
                    pending[vid] = {"status": "unavailable", "last_scan_at": iso(now), "reason": str(e)}
                    summary["unavailable"] += 1
                    processed += 1
                    continue
                pending[vid] = {"status": "ok", "last_scan_at": iso(now), "scan_comment_count": cc}
                summary["videos_scanned"] += 1
                processed += 1
            if len(pending) >= SAVE_STATE_EVERY:
                commit()
    except QuotaExhausted:
        summary["stopped"] = "quota"
        log.warning("Hết quota YouTube. Phần còn lại sẽ làm tiếp ở lần chạy sau.")
    except BudgetExceeded as e:
        summary["stopped"] = str(e)
        log.warning("Dừng vì %s. Phần còn lại sẽ làm tiếp ở lần chạy sau.", e)
    finally:
        commit()

    summary.update(new_comments=col.new_top, new_replies=col.new_replies, units_used=budget.units_used,
                   remaining=len(queue) - processed)
    log.info("fetch_comments: %s", summary)
    return summary