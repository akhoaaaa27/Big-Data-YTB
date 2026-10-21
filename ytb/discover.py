"""Task discover_videos: tìm video nhạc AI mới (phần 3 ngày gần nhất + lùi dần backfill)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import config
from .storage import iso, known_video_ids, load_state, parse_iso, save_state, write_rows
from .youtube_client import Budget, BudgetExceeded, QuotaExhausted, build_client, call

log = logging.getLogger(__name__)
STATE_FILE = "backfill_state.json"   # cùng định dạng file cũ: {"next_before": "..."}


def search_window(client, budget, published_after, published_before=None) -> dict[str, str]:
    """Tìm theo từng từ khóa trong một khoảng thời gian. Trả về {video_id: từ khóa tìm thấy đầu tiên}."""
    found: dict[str, str] = {}
    for kw in config.KEYWORDS:
        params = dict(
            part="snippet", q=kw, type="video", order="relevance",
            maxResults=config.VIDEOS_PER_KEYWORD,
            regionCode=config.REGION_CODE,
            relevanceLanguage=config.RELEVANCE_LANGUAGE,
            publishedAfter=iso(published_after),
        )
        if published_before:
            params["publishedBefore"] = iso(published_before)
        res = call(client.search().list(**params), budget, kind="search")
        for item in res.get("items", []):
            found.setdefault(item["id"]["videoId"], kw)
    return found


def video_row(item: dict, keyword: str | None, now: datetime, source: str = "api") -> dict:
    """Biến một kết quả videos.list thành một dòng của bảng videos."""
    sn = item.get("snippet", {})
    return {
        "video_id": item["id"],
        "keyword": keyword,
        "title": sn.get("title"),
        "channel_id": sn.get("channelId"),
        "channel_title": sn.get("channelTitle"),
        "description": sn.get("description"),
        "tags": sn.get("tags") or [],
        "published_at": sn.get("publishedAt"),
        "category_id": sn.get("categoryId"),
        "default_language": sn.get("defaultLanguage"),
        "default_audio_language": sn.get("defaultAudioLanguage"),
        "duration": item.get("contentDetails", {}).get("duration"),
        "contains_synthetic_media": item.get("status", {}).get("containsSyntheticMedia"),
        "source": source,
        "fetched_at": iso(now),
    }


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def store_new_videos(client, budget, found: dict[str, str], known: set[str], now: datetime) -> int:
    """Lấy siêu dữ liệu đầy đủ cho video chưa có và ghi vào Staging. Trả về số video mới."""
    new_ids = [v for v in found if v not in known]
    stored = 0
    for batch in chunks(new_ids, 50):
        res = call(client.videos().list(part="snippet,status,contentDetails", id=",".join(batch)), budget)
        rows = [video_row(it, found.get(it["id"]), now) for it in res.get("items", [])]
        write_rows("videos", rows, now.strftime("%Y-%m-%d"))
        known.update(r["video_id"] for r in rows)
        stored += len(rows)
    return stored


def run_discover(client=None, now: datetime | None = None, budget: Budget | None = None) -> dict:
    client = client or build_client()
    now = now or datetime.now(timezone.utc)
    budget = budget or Budget(config.QUOTA_BUDGET_OTHER, config.SEARCH_CALL_BUDGET)
    known = known_video_ids()
    summary = {"new_recent": 0, "new_backfill": 0, "windows_done": 0, "stopped": None}

    try:
        # A. Video đăng trong DAYS_BACK ngày gần nhất
        found = search_window(client, budget, now - timedelta(days=config.DAYS_BACK))
        summary["new_recent"] = store_new_videos(client, budget, found, known, now)

        # B. Lùi dần về quá khứ, tiến độ lưu sau khi đã ghi dữ liệu
        state = load_state(STATE_FILE, {})
        cursor = parse_iso(state["next_before"]) if "next_before" in state else now - timedelta(days=config.DAYS_BACK)
        for _ in range(config.WINDOWS_PER_RUN):
            if cursor <= config.BACKFILL_STOP:
                log.info("Backfill đã chạm mốc %s", config.BACKFILL_STOP.date())
                break
            start = max(cursor - timedelta(days=config.WINDOW_DAYS), config.BACKFILL_STOP)
            found = search_window(client, budget, start, cursor)
            summary["new_backfill"] += store_new_videos(client, budget, found, known, now)
            save_state(STATE_FILE, {"next_before": iso(start)})
            cursor = start
            summary["windows_done"] += 1
    except QuotaExhausted:
        summary["stopped"] = "quota"
        log.warning("Hết quota YouTube. Dừng sớm, lần chạy sau làm tiếp.")
    except BudgetExceeded as e:
        summary["stopped"] = str(e)
        log.warning("Dừng vì %s.", e)

    summary["search_calls"] = budget.search_used
    summary["units_used"] = budget.units_used
    log.info("discover_videos: %s", summary)
    return summary