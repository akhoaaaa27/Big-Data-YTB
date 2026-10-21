"""Lọc nhạc AI tự động từ Core; không cần điền nhãn tay.

python -m ytb.curate          # xuất data/curated và báo cáo data/review/videos.csv
python -m ytb.curate export   # chỉ xuất báo cáo phân loại tự động
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import config

METHOD = "automatic_audio_metadata_v1"
FIELDS = ["video_id", "url", "title", "channel_title", "n_comments", "decision", "reason", "evidence"]
MUSIC_AI = re.compile(
    r"\b(?:nhac|am nhac|bai hat|ca khuc|giong hat|giong ca|ca si|hat|cover|remix|bolero|rumba) "
    r"(?:(?:do|bang|boi|tu|viet|cover) )?ai\b"
    r"|\bai[ -]?(?:music|song|cover|vocal|singing|singer|remix)\b"
    r"|\b(?:music|song|vocals?|voice|singing|cover) (?:generated |created |made )?(?:by |with |using )?ai\b"
    r"|\b(?:cover|hat|sang tac|tao nhac) (?:bang |boi |do )?tri tue nhan tao\b"
)
PROVIDER = re.compile(r"\b(?:suno(?:ai)?|udio)\b")
AUDIO_CREDIT = re.compile(
    r"\b(?:nhac|am nhac|ca si|trinh bay|giong hat|sang tac|music|song|vocals?|voice|created|generated|produced|made)"
    r"[^\n.!?]{0,60}\b(?:suno(?:ai)?|udio)\b"
)
NON_MUSIC = re.compile(
    r"\b(?:huong dan|tutorial|how to|review phim|tom tat phim|gameplay|cover mechanics|rts)\b"
    r"|\bcach (?:su dung|dung|tao|lam)\b|\b\d+ buoc\b"
    r"|\b(?:meo|thu thuat) (?:tao nhac|suno|udio)\b"
    r"|\breview (?:cong cu|suno|udio)\b|\btest (?:nhac|loa|am thanh)\b"
    r"|\b(?:music|song) generators?\b|\bai slop\b|\bda bi cam\b|\bchuyen gi tiep theo\b"
    r"|\bbi mat tao nhac\b|\b(?:podcast|text to speech|voice assistant)\b"
    r"|\btao (?:mv|video)\b.*\b(?:bang ai|mien phi|don gian)\b"
    r"|कैसे|बनाएं|बनाना"
)
DENIAL = re.compile(
    r"\b(?:khong|ko) (?:su dung|dung|phai|tao bang|tao boi) (?:nhac )?(?:ai|suno|udio)\b"
    r"|\b(?:no ai|not ai|without ai|non ai)\b"
)
VISUAL = re.compile(r"\b(?:ai art|ai video|ai animation|ai image|cover art|hinh anh ai|anh ai|video ai)\b")


def normalize(value) -> str:
    text = html.unescape(str(value or "")).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\ba\s*\.\s*i\.?", "ai", text)
    text = unicodedata.normalize("NFD", text)
    # 'ái/ải/ãi/ài/ại' trong tên bài không phải ký hiệu AI.
    text = re.sub(r"\ba[\u0300-\u036f]+i\b", "tuvietai", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("đ", "d")
    return re.sub(r"[^\S\n]+", " ", text.replace("_", " ")).strip()


def classify(video: dict) -> tuple[str, str, str]:
    """Không lấy keyword, tên kênh hoặc cờ synthetic làm bằng chứng nhạc AI."""
    title = normalize(video.get("title"))
    description = normalize(video.get("description"))
    channel = normalize(video.get("channel_title"))
    if len(channel) >= 6:
        # Không coi tên kênh chèn trong phần quảng bá là lời công bố về âm thanh.
        description = description.replace(channel, " ")
    tags = normalize(" ".join(video.get("tags") or []))
    negative = NON_MUSIC.search(title)
    if negative:
        return "exclude", "Nội dung hướng dẫn/game/review/test, không phải tác phẩm nhạc AI", negative.group()
    denial = DENIAL.search(title + "\n" + description)
    if denial:
        return "uncertain", "Metadata có phủ định AI; không tự đưa vào tập phân tích", denial.group()
    if str(video.get("category_id")) in ("20", "25", "27", "28"):
        return "uncertain", "Video thuộc nhóm game/tin tức/giáo dục/công nghệ; không tự coi là tác phẩm âm nhạc", ""
    # Đại từ 'ai' trong tên bài không phải ký hiệu trí tuệ nhân tạo. Vẫn giữ
    # tín hiệu riêng như '- AI Cover' ở cuối cùng một tiêu đề.
    title = re.sub(r"\bai (?=dua\b|oi\b|cung\b|la\b|moi\b|dang\b|biet\b|se\b|cho\b|con\b|ve\b|yeu\b)", "nguoi nao ", title)
    title = re.sub(r"\b(cho|vi|nho|thuong|doi|ben|tim|cung|yeu|theo) ai\b", r"\1 nguoi nao", title)
    # Hashtag là tín hiệu yếu dù được đặt ở tiêu đề; không duyệt clip nhảy chỉ
    # vì caption gắn #aicover. Tên bài kèm nhãn 'AI Cover' vẫn được nhận diện.
    title = re.sub(r"#\w+", " ", title)
    # Hình ảnh/MV dùng AI không đồng nghĩa phần âm nhạc cũng do AI tạo.
    visual = VISUAL.search(title)
    title_audio = MUSIC_AI.search(title)
    if visual and not title_audio and not PROVIDER.search(title):
        return "uncertain", "Chỉ thấy dấu hiệu AI ở hình ảnh/video", visual.group()
    if title_audio:
        return "include", "Tiêu đề tự công bố nhạc/giọng hát/cover AI", title_audio.group()
    provider = PROVIDER.search(title)
    if provider:
        return "include", "Tiêu đề nêu công cụ tạo âm nhạc Suno/Udio", provider.group()
    singing_label = re.search(r"[\[(]\s*ai hat\s*[\])]", title)
    if singing_label:
        return "include", "Tiêu đề có nhãn giọng hát AI", singing_label.group()
    # Credits trong mô tả: tách từng dòng, không ghép câu ca từ với tên công cụ.
    for line in description.splitlines():
        # Bỏ quảng bá tên kênh, danh sách hashtag/từ khóa và đoạn bình luận về AI.
        if len(line) > 350 or re.search(r"\b(?:kenh|channel|subscribe|follow|dang ky|youtube|nguon tham khao|tags|tu khoa)\b|#", line):
            continue
        credit = AUDIO_CREDIT.search(line)
        if credit:
            return "include", "Mô tả ghi nguồn tạo âm nhạc/giọng hát bằng Suno/Udio", credit.group()
        desc_audio = MUSIC_AI.search(line)
        if desc_audio:
            return "include", "Mô tả tự công bố phần âm nhạc/giọng hát dùng AI", desc_audio.group()
    if PROVIDER.search(tags) or MUSIC_AI.search(tags) or video.get("contains_synthetic_media"):
        return "uncertain", "Chỉ có tags/cờ synthetic, chưa đủ bằng chứng về phần âm thanh", ""
    return "uncertain", "Không có bằng chứng rõ về nhạc AI trong metadata", ""


def review_path() -> Path:
    return config.DATA_DIR / "review" / "videos.csv"


def _load():
    source = config.DATA_DIR / "core"
    videos = ds.dataset(source / "videos", format="parquet").to_table().to_pylist()
    ids = [v["video_id"] for v in videos]
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Core videos phải có video_id duy nhất và không rỗng")
    counts = Counter()
    if (source / "comments").exists():
        for batch in ds.dataset(source / "comments", format="parquet").to_batches(columns=["video_id"]):
            counts.update(batch.column(0).to_pylist())
    rows = []
    for video in videos:
        decision, reason, evidence = classify(video)
        vid = video["video_id"]
        rows.append(dict(video_id=vid, url=f"https://www.youtube.com/watch?v={vid}",
                         title=video.get("title") or "", channel_title=video.get("channel_title") or "",
                         n_comments=counts[vid], decision=decision, reason=reason, evidence=evidence))
    return sorted(rows, key=lambda r: (-r["n_comments"], r["video_id"]))


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def export_review() -> dict:
    rows = _load()
    _write_csv(review_path(), rows)
    return {"report": str(review_path()), "decisions": dict(Counter(r["decision"] for r in rows))}


def run_auto() -> dict:
    rows = _load()
    included = sorted(r["video_id"] for r in rows if r["decision"] == "include")
    accepted = pa.array(included, type=pa.string())
    source = config.DATA_DIR / "core"
    target = config.DATA_DIR / "curated"
    backup = config.DATA_DIR / "curated.old"
    if backup.exists():
        raise ValueError("Còn curated.old từ lần chạy bị gián đoạn; cần kiểm tra/khôi phục trước")
    tmp = Path(tempfile.mkdtemp(prefix="curated-building-", dir=config.DATA_DIR))
    summary = {"method": METHOD, "included_video_ids": included,
               "decisions": {v: sum(r["decision"] == v for r in rows) for v in ("include", "exclude", "uncertain")},
               "generated_at": datetime.now(timezone.utc).isoformat()}
    try:
        for name in ("videos", "comments", "video_snapshots"):
            folder = source / name
            if not folder.exists():
                if name == "comments":
                    raise ValueError("Core chưa có bảng comments")
                continue
            dataset = ds.dataset(folder, format="parquet")
            out = tmp / name
            out.mkdir()
            before = after = 0
            with pq.ParquetWriter(out / "part-00000.parquet", dataset.schema, compression="zstd") as writer:
                for batch in dataset.to_batches():
                    before += batch.num_rows
                    table = pa.Table.from_batches([batch])
                    table = table.filter(pc.is_in(table["video_id"], value_set=accepted))
                    after += table.num_rows
                    writer.write_table(table)
            summary[name] = {"before": before, "kept": after, "removed": before - after}
        _write_csv(tmp / "classification.csv", rows)
        _write_csv(tmp / "uncertain.csv", [r for r in rows if r["decision"] == "uncertain"])
        (tmp / "review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if target.exists():
            target.rename(backup)
        try:
            tmp.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    _write_csv(review_path(), rows)
    return summary


# Tương thích lệnh apply cũ; không còn đọc/yêu cầu quyết định bằng tay.
apply_review = run_auto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="auto", choices=("auto", "export", "apply"))
    args = parser.parse_args()
    result = export_review() if args.command == "export" else run_auto()
    print(json.dumps({k: v for k, v in result.items() if k != "included_video_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
