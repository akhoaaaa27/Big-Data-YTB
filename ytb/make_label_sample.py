"""Lấy mẫu bình luận từ data/curated (nhạc AI đã lọc tự động) để nhóm gán nhãn cảm xúc.

Cách chia: một nhóm bình luận CHUNG (mọi người cùng chấm, dùng để đo độ đồng thuận giữa người chấm)
và mỗi người một nhóm RIÊNG (không trùng nhau). Mỗi video tối đa --cap bình luận để mẫu trải đều.

Chạy:  docker compose --profile debug run --rm airflow-cli python -m ytb.make_label_sample
Kết quả trong data/labeling/:
  phieu_<tên>.csv     phát cho từng người chấm (không có thông tin về mô hình hay lượt like)
  sample_master.csv   bản tổng của nhóm trưởng: ai được giao bình luận nào
Chạy lại sẽ KHÔNG ghi đè (tránh mất nhãn đã chấm) trừ khi thêm --force.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from . import config
from .curate import METHOD

LABELS = ("tich_cuc", "tieu_cuc", "trung_lap", "khong_xac_dinh")
_BR = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def label_dir() -> Path:
    return config.DATA_DIR / "labeling"


def display_text(raw: str) -> str:
    """Bản dễ đọc cho người chấm: giữ nguyên chữ hoa, emoji; chỉ bỏ thẻ HTML."""
    t = _BR.sub("\n", raw or "")
    return html.unescape(_TAG.sub("", t)).strip()


def load_candidates() -> list[dict]:
    folder = config.DATA_DIR / "curated" / "comments"
    if not folder.exists():
        raise SystemExit("Chưa có dữ liệu lọc tự động. Chạy python -m ytb.curate trước.")
    cols = ["comment_id", "video_id", "is_reply", "text_raw", "text_clean", "is_modelable", "source"]
    rows = ds.dataset(str(folder), format="parquet").to_table(columns=cols).to_pylist()
    report = config.DATA_DIR / "curated" / "review.json"
    if not report.exists():
        raise SystemExit("Thiếu báo cáo lọc tự động. Chạy python -m ytb.curate trước.")
    metadata = json.loads(report.read_text(encoding="utf-8"))
    approved = set(metadata.get("included_video_ids", []))
    if metadata.get("method") != METHOD or not approved or any(r["video_id"] not in approved for r in rows):
        raise SystemExit("Dữ liệu và báo cáo lọc không khớp. Chạy python -m ytb.curate trước khi lấy mẫu.")
    rows = [r for r in rows if r["text_clean"]]          # bỏ bình luận rỗng sau khi làm sạch
    vfolder = config.DATA_DIR / "curated" / "videos"
    info: dict = {}
    if vfolder.exists():
        v = ds.dataset(str(vfolder), format="parquet").to_table(columns=["video_id", "title", "keyword"]).to_pylist()
        info = {x["video_id"]: x for x in v}
    for r in rows:
        meta = info.get(r["video_id"], {})
        r["video_title"] = meta.get("title") or ""
        r["keyword"] = meta.get("keyword") or ""
    return rows


def draw(rows: list[dict], n: int, cap: int, rng: random.Random) -> list[dict]:
    """Lấy n bình luận ngẫu nhiên, mỗi video tối đa cap bình luận."""
    pool = sorted(rows, key=lambda r: r["comment_id"])    # sắp xếp trước để cùng seed luôn ra cùng mẫu
    rng.shuffle(pool)
    per_video: dict[str, int] = defaultdict(int)
    out: list[dict] = []
    for r in pool:
        if per_video[r["video_id"]] >= cap:
            continue
        per_video[r["video_id"]] += 1
        out.append(r)
        if len(out) == n:
            return out
    raise SystemExit(f"Chỉ lấy được {len(out)}/{n} bình luận với giới hạn {cap} bình luận mỗi video. "
                     f"Tăng --cap hoặc giảm số lượng.")


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:     # utf-8-sig để Excel/Sheets đọc đúng tiếng Việt
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run(names: list[str], shared: int, per_person: int, cap: int, seed: int, force: bool = False) -> dict:
    out = label_dir()
    master = out / "sample_master.csv"
    if master.exists() and not force:
        raise SystemExit(f"{master} đã tồn tại. Chạy lại sẽ đổi mẫu và làm mất nhãn đã chấm. Thêm --force nếu thật sự muốn.")
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    picked = draw(load_candidates(), shared + per_person * len(names), cap, rng)
    shared_rows, rest = picked[:shared], picked[shared:]

    master_rows, summary = [], {}
    for i, name in enumerate(names):
        own = rest[i * per_person:(i + 1) * per_person]
        sheet = [(r, "chung") for r in shared_rows] + [(r, "rieng") for r in own]
        rng.shuffle(sheet)                                # mỗi người một thứ tự khác nhau
        _write(out / f"phieu_{name}.csv",
               ["stt", "comment_id", "video_title", "binh_luan", "cam_xuc", "chu_de_ghi_chu", "ghi_chu"],
               [[k, r["comment_id"], r["video_title"], display_text(r["text_raw"]), "", "", ""]
                for k, (r, _) in enumerate(sheet, 1)])
        summary[name] = len(sheet)
        master_rows += [[name, grp, r["comment_id"], r["video_id"], r["video_title"], r["keyword"],
                         r["is_reply"], r["source"], r["is_modelable"], display_text(r["text_raw"])] for r, grp in sheet]

    _write(master, ["nguoi_cham", "nhom", "comment_id", "video_id", "video_title", "keyword",
                    "is_reply", "source", "is_modelable", "binh_luan"], master_rows)
    summary["_unique_comments"] = len(picked)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["Bao", "Minh", "Tu", "Khoa"], help="tên người chấm (không dấu, dùng đặt tên file)")
    ap.add_argument("--shared", type=int, default=50, help="số bình luận chung cho mọi người")
    ap.add_argument("--per-person", type=int, default=62, help="số bình luận riêng mỗi người")
    ap.add_argument("--cap", type=int, default=3, help="tối đa bao nhiêu bình luận mỗi video")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    s = run(a.names, a.shared, a.per_person, a.cap, a.seed, a.force)
    print(f"Đã tạo {s.pop('_unique_comments')} bình luận khác nhau trong {label_dir()}")
    for name, n in s.items():
        print(f"  phieu_{name}.csv: {n} dòng")


if __name__ == "__main__":
    main()
