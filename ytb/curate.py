"""Duyệt độ liên quan của video, lọc Core sang data/curated mà không sửa dữ liệu gốc.

python -m ytb.curate export  # tạo/cập nhật data/review/videos.csv
python -m ytb.curate apply   # chỉ giữ decision=include; để trống = chưa duyệt
"""

from __future__ import annotations

import argparse
import csv
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

FIELDS = ["video_id", "url", "title", "channel_title", "n_comments", "suggestion", "reason", "decision", "note"]


def review_path() -> Path:
    return config.DATA_DIR / "review" / "videos.csv"


def suggest(video: dict) -> tuple[str, str]:
    """Chỉ gợi ý để ưu tiên duyệt, không dùng keyword tìm kiếm làm bằng chứng."""
    def norm(s):
        s = unicodedata.normalize("NFD", str(s or "").casefold())
        return "".join(c for c in s if not unicodedata.combining(c)).replace("đ", "d")

    title = norm(video.get("title"))
    text = " ".join([title, norm(video.get("description")), norm(" ".join(video.get("tags") or []))])
    if re.search(r"huong dan|cach (?:tao|lam)|\btutorial\b|\bhow to\b|\d+ buoc|review phim|tom tat phim", title):
        return "likely_other", "Có dấu hiệu hướng dẫn/review; cần phân biệt với tác phẩm âm nhạc"
    if re.search(r"\bsuno\b|\budio\b|\baimusic\b|\bsunoai\b|\bai[ -]?(?:music|cover|generated)\b|\bcover ai\b|\bnhac (?:a\.?i\.?|tri tue nhan tao)\b", text):
        return "likely_ai_music", "Metadata có dấu hiệu nhạc AI; vẫn cần xác nhận tác phẩm thực tế"
    return "review", "Chưa đủ bằng chứng; từ khóa tìm kiếm và cờ synthetic không chứng minh là nhạc AI"


def read_decisions() -> dict[str, dict]:
    path = review_path()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"video_id", "decision"}.issubset(reader.fieldnames or []):
            raise ValueError("File duyệt phải có cột video_id và decision")
        result = {}
        for row in reader:
            vid = row["video_id"].strip()
            decision = row["decision"].strip().lower()
            if not vid or vid in result or decision not in ("", "include", "exclude"):
                raise ValueError(f"Dòng duyệt không hợp lệ hoặc trùng video_id: {vid!r}")
            result[vid] = {**row, "decision": decision}
        return result


def export_review() -> dict:
    source = config.DATA_DIR / "core" / "videos"
    videos = ds.dataset(source, format="parquet").to_table().to_pylist()
    ids = [v["video_id"] for v in videos]
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Core videos phải có video_id duy nhất và không rỗng")
    previous = read_decisions()
    counts = Counter()
    comments = config.DATA_DIR / "core" / "comments"
    if comments.exists():
        for batch in ds.dataset(comments, format="parquet").to_batches(columns=["video_id"]):
            counts.update(batch.column(0).to_pylist())
    path = review_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for video in videos:
        vid = video["video_id"]
        suggestion, reason = suggest(video)
        old = previous.get(vid, {})
        rows[vid] = dict(video_id=vid, url=f"https://www.youtube.com/watch?v={vid}",
                         title=video.get("title") or "", channel_title=video.get("channel_title") or "", n_comments=counts[vid],
                         suggestion=suggestion, reason=reason, decision=old.get("decision", ""), note=old.get("note", ""))
    # Giữ cả quyết định về video tạm vắng mặt trong Core để không mất công duyệt.
    for vid, row in previous.items():
        rows.setdefault(vid, {k: row.get(k, "") for k in FIELDS})
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[v] for v in sorted(rows, key=lambda v: (-counts[v], v)))
    os.replace(tmp, path)
    return {"review_file": str(path), "videos": len(videos), "pending": sum(not rows[v]["decision"] for v in ids)}


def apply_review() -> dict:
    decisions = read_decisions()
    if not decisions:
        raise ValueError("Chưa có danh sách duyệt. Chạy python -m ytb.curate export trước")
    source = config.DATA_DIR / "core"
    videos = ds.dataset(source / "videos", format="parquet").to_table()
    ids = videos.column("video_id").to_pylist()
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Core videos phải có video_id duy nhất và không rỗng")
    approved = sorted(v for v in ids if decisions.get(v, {}).get("decision") == "include")
    if not approved:
        raise ValueError("Chưa có video nào được duyệt include; giữ nguyên dữ liệu hiện có")
    accepted = pa.array(approved, type=pa.string())
    target = config.DATA_DIR / "curated"
    backup = config.DATA_DIR / "curated.old"
    if backup.exists():
        raise ValueError("Còn curated.old từ lần chạy bị gián đoạn; cần kiểm tra/khôi phục trước")
    tmp = Path(tempfile.mkdtemp(prefix="curated-building-", dir=config.DATA_DIR))
    summary = {"approved_videos": len(approved), "pending_videos": sum(not decisions.get(v, {}).get("decision") for v in ids),
               "approved_video_ids": approved, "generated_at": datetime.now(timezone.utc).isoformat()}
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
            # Đọc từng batch để không nạp toàn bộ bình luận vào RAM.
            with pq.ParquetWriter(out / "part-00000.parquet", dataset.schema, compression="zstd") as writer:
                for batch in dataset.to_batches():
                    before += batch.num_rows
                    table = pa.Table.from_batches([batch])
                    table = table.filter(pc.is_in(table["video_id"], value_set=accepted))
                    after += table.num_rows
                    writer.write_table(table)
            summary[name] = {"before": before, "kept": after, "removed": before - after}
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
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "apply"))
    args = parser.parse_args()
    result = export_review() if args.command == "export" else apply_review()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
