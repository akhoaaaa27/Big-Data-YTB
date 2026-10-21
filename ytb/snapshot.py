"""Task snapshot_stats: ghi nối thêm view/like/comment_count của mọi video đã biết, mỗi ngày một dòng."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import config
from .discover import chunks, video_row
from .storage import iso, known_video_ids, snapshot_video_ids_on, video_meta, write_rows
from .youtube_client import Budget, BudgetExceeded, QuotaExhausted, build_client, call

log = logging.getLogger(__name__)


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None   # ví dụ video ẩn lượt thích


def run_snapshot(client=None, now: datetime | None = None, budget: Budget | None = None) -> dict:
    client = client or build_client()
    now = now or datetime.now(timezone.utc)
    budget = budget or Budget(config.QUOTA_BUDGET_OTHER)
    day = now.strftime("%Y-%m-%d")

    meta = video_meta()                      # video cũ từ CSV chưa có channel_id... sẽ được bổ sung ở đây
    already_today = snapshot_video_ids_on(day)
    todo = sorted(v for v in known_video_ids() if v not in already_today)
    summary = {"snapshots": 0, "metadata_enriched": 0, "unavailable": 0, "stopped": None}

    try:
        for batch in chunks(todo, 50):
            res = call(client.videos().list(part="snippet,statistics,status,contentDetails", id=",".join(batch)), budget)
            snaps, enrich = [], []
            returned = set()
            for it in res.get("items", []):
                returned.add(it["id"])
                st = it.get("statistics", {})
                snaps.append({
                    "video_id": it["id"], "snapshot_date": day, "snapshot_at": iso(now),
                    "view_count": _to_int(st.get("viewCount")),
                    "like_count": _to_int(st.get("likeCount")),
                    "comment_count": _to_int(st.get("commentCount")),
                })
                if not meta.get(it["id"], {}).get("complete"):
                    enrich.append(video_row(it, meta.get(it["id"], {}).get("keyword"), now))
            write_rows("video_snapshots", snaps, day)
            write_rows("videos", enrich, day)
            summary["snapshots"] += len(snaps)
            summary["metadata_enriched"] += len(enrich)
            summary["unavailable"] += len(set(batch) - returned)
    except QuotaExhausted:
        summary["stopped"] = "quota"
        log.warning("Hết quota YouTube khi chụp snapshot.")
    except BudgetExceeded as e:
        summary["stopped"] = str(e)

    summary["units_used"] = budget.units_used
    log.info("snapshot_stats: %s", summary)
    return summary