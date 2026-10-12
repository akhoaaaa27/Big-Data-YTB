import os
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# ====== CẤU HÌNH ======
KEYWORDS = [
    "nhạc AI", "bài hát do AI sáng tác", "nhạc Suno AI",
    "AI cover", "AI hát", "nhạc AI Việt Nam",
]
VIDEOS_PER_KEYWORD = 50
COMMENTS_PER_VIDEO = 200
DAYS_BACK = 3                     # phần video mới (mỗi ngày)
WINDOW_DAYS = 7                   # độ dài mỗi cửa sổ lùi về quá khứ
WINDOWS_PER_RUN = 5               # số cửa sổ quá khứ mỗi lần chạy
BACKFILL_STOP = datetime(2024, 1, 1, tzinfo=timezone.utc)

# --- Lấy lại bình luận cho video thiếu ---
MIN_AGE_DAYS = 7                  # video phải đăng ít nhất N ngày mới kiểm tra lại
RECHECK_COOLDOWN_DAYS = 7         # cách nhau tối thiểu N ngày giữa 2 lần kiểm tra
MAX_TRIES = 3                     # tối đa số lần kiểm tra lại mỗi video
MIN_GAP = 5                       # chỉ lấy lại nếu thiếu ít nhất N bình luận
MAX_RECHECK_PER_RUN = 100         # số video kiểm tra lại mỗi lần chạy
# ======================

VIDEO_FILE = "videos_nhac_ai.csv"
COMMENT_FILE = "comments_nhac_ai.csv"
STATE_FILE = "backfill_state.json"
RECHECK_FILE = "recheck_state.json"
VIDEO_FIELDS = ["video_id", "keyword", "title", "channel", "published",
                "views", "likes", "comment_count"]
COMMENT_FIELDS = ["comment_id", "video_id", "author", "text", "likes", "published"]
TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def iso(dt):
    return dt.strftime(TIME_FMT)


def parse(s):
    return datetime.strptime(s, TIME_FMT).replace(tzinfo=timezone.utc)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    if os.path.exists(path):
        shutil.copy(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def save_all():
    save_csv(VIDEO_FILE, list(videos.values()), VIDEO_FIELDS)
    save_csv(COMMENT_FILE, comments, COMMENT_FIELDS)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def comment_key(r):
    return (r["video_id"], r["author"], r["published"], r["text"].strip())


# ---- Đọc dữ liệu đã có ----
videos = {r["video_id"]: r for r in load_csv(VIDEO_FILE)}
comments = load_csv(COMMENT_FILE)
seen = {comment_key(r) for r in comments}
recheck = load_json(RECHECK_FILE, {})
print(f"Đã có sẵn: {len(videos)} video, {len(comments)} bình luận")


def fetch_comments(vid):
    """Lấy bình luận gốc của 1 video, bỏ qua bình luận đã có.
    Trả về số bình luận đã duyệt, hoặc -1 nếu video tắt bình luận."""
    token, got = None, 0
    while got < COMMENTS_PER_VIDEO:
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=vid, maxResults=100,
                order="relevance", textFormat="plainText", pageToken=token,
            ).execute()
        except HttpError as e:
            msg = str(e)
            if "commentsDisabled" in msg:
                return -1
            if "quotaExceeded" in msg:
                raise                     # hết quota: dừng hẳn, finally sẽ lưu dữ liệu
            break
        for c in res["items"]:
            top = c["snippet"]["topLevelComment"]["snippet"]
            row = {
                "comment_id": c["id"],
                "video_id": vid,
                "author": top["authorDisplayName"],
                "text": top["textDisplay"],
                "likes": top["likeCount"],
                "published": top["publishedAt"],
            }
            if comment_key(row) not in seen:
                seen.add(comment_key(row))
                comments.append(row)
            got += 1
        token = res.get("nextPageToken")
        if not token:
            break
    return got


def mark_checked(vid, got):
    st = recheck.get(vid, {"tries": 0})
    st["tries"] = st.get("tries", 0) + 1
    st["last"] = iso(datetime.now(timezone.utc))
    if got == -1:
        st["done"] = True                 # tắt bình luận, khỏi kiểm tra lại
    recheck[vid] = st


def search_window(after, before=None):
    found = {}
    for kw in KEYWORDS:
        params = dict(
            part="snippet", q=kw, type="video", maxResults=VIDEOS_PER_KEYWORD,
            regionCode="VN", relevanceLanguage="vi", order="relevance",
            publishedAfter=iso(after),
        )
        if before:
            params["publishedBefore"] = iso(before)
        res = youtube.search().list(**params).execute()
        for item in res["items"]:
            found.setdefault(item["id"]["videoId"], kw)
    return found


def process(found):
    """Cập nhật thống kê video; lấy bình luận cho video MỚI."""
    new_ids = [i for i in found if i not in videos]
    print(f"  Tìm thấy {len(found)} video, {len(new_ids)} video MỚI")

    ids = list(found)
    for i in range(0, len(ids), 50):
        res = youtube.videos().list(
            part="snippet,statistics", id=",".join(ids[i:i + 50])
        ).execute()
        for v in res["items"]:
            s = v["statistics"]
            old = videos.get(v["id"], {})
            videos[v["id"]] = {
                "video_id": v["id"],
                "keyword": old.get("keyword", found[v["id"]]),
                "title": v["snippet"]["title"],
                "channel": v["snippet"]["channelTitle"],
                "published": v["snippet"]["publishedAt"],
                "views": s.get("viewCount", 0),
                "likes": s.get("likeCount", 0),
                "comment_count": s.get("commentCount", 0),
            }

    for n, vid in enumerate(new_ids, 1):
        got = fetch_comments(vid)
        mark_checked(vid, got)
        print(f"  [{n}/{len(new_ids)}] {vid}: {max(got, 0)} bình luận")


def recheck_comments():
    """Lấy bổ sung bình luận cho video cũ đang thiếu."""
    now = datetime.now(timezone.utc)
    have = Counter(c["video_id"] for c in comments)
    candidates = []
    for vid, v in videos.items():
        st = recheck.get(vid, {})
        if st.get("done") or st.get("tries", 0) >= MAX_TRIES:
            continue
        try:
            if now - parse(v["published"]) < timedelta(days=MIN_AGE_DAYS):
                continue
            if st.get("last") and now - parse(st["last"]) < timedelta(days=RECHECK_COOLDOWN_DAYS):
                continue
            total = int(v["comment_count"] or 0)
        except (ValueError, KeyError):
            continue
        h = have[vid]
        if h >= COMMENTS_PER_VIDEO or total - h < MIN_GAP:
            continue
        candidates.append((total - h, vid))

    candidates.sort(reverse=True)
    batch = [vid for _, vid in candidates[:MAX_RECHECK_PER_RUN]]
    print(f"  {len(candidates)} video đang thiếu bình luận, kiểm tra lại {len(batch)} video")

    before_total = len(comments)
    for n, vid in enumerate(batch, 1):
        before = len(comments)
        got = fetch_comments(vid)
        mark_checked(vid, got)
        print(f"  [{n}/{len(batch)}] {vid}: +{len(comments) - before} bình luận mới")
    print(f"  Bổ sung được {len(comments) - before_total} bình luận")


def load_cursor(default):
    if os.path.exists(STATE_FILE):
        return parse(load_json(STATE_FILE, {})["next_before"])
    return default


def save_cursor(dt):
    save_json(STATE_FILE, {"next_before": iso(dt)})


now = datetime.now(timezone.utc)
try:
    # ---- A. Video mới (DAYS_BACK ngày gần nhất) ----
    print(f"== A. Video {DAYS_BACK} ngày gần nhất ==")
    process(search_window(now - timedelta(days=DAYS_BACK)))
    save_all()
    save_json(RECHECK_FILE, recheck)

    # ---- C. Lấy lại bình luận cho video đang thiếu ----
    print("== C. Bổ sung bình luận cho video thiếu ==")
    recheck_comments()
    save_all()
    save_json(RECHECK_FILE, recheck)

    # ---- B. Lùi dần về quá khứ ----
    cursor = load_cursor(now - timedelta(days=DAYS_BACK))
    for _ in range(WINDOWS_PER_RUN):
        if cursor <= BACKFILL_STOP:
            print("== B. Đã lùi tới mốc dừng, xong phần video cũ ==")
            break
        start = max(cursor - timedelta(days=WINDOW_DAYS), BACKFILL_STOP)
        print(f"== B. Video đăng từ {start:%Y-%m-%d} đến {cursor:%Y-%m-%d} ==")
        process(search_window(start, cursor))
        save_all()                      # lưu dữ liệu trước,
        save_json(RECHECK_FILE, recheck)
        save_cursor(start)              # rồi mới ghi tiến độ
        cursor = start
finally:
    save_all()
    save_json(RECHECK_FILE, recheck)
    print(f"XONG: tổng {len(videos)} video, {len(comments)} bình luận")