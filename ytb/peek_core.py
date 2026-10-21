"""Xem nhanh tầng Core để kiểm tra việc làm sạch: phân bố số từ, mẫu bình luận bị loại, từ phổ biến nhất.

Chạy:  docker compose --profile debug run --rm airflow-cli python -m ytb.peek_core
"""

from __future__ import annotations

import random
from collections import Counter

import pyarrow.dataset as ds

from . import config

BUCKETS = [(0, 0, "0 từ"), (1, 1, "1 từ"), (2, 2, "2 từ"), (3, 3, "3 từ"), (4, 5, "4-5 từ"), (6, 10, "6-10 từ"), (11, 10**9, "11+ từ")]


def main(seed: int = 7, n_sample: int = 15) -> None:
    folder = config.DATA_DIR / "core" / "comments"
    if not folder.exists():
        print("Chưa có Core. Chạy python -m ytb.spark_core trước.")
        return
    t = ds.dataset(str(folder), format="parquet").to_table(
        columns=["text_raw", "text_clean", "tokens", "n_tokens", "is_reply"]).to_pydict()
    n = len(t["n_tokens"])
    rng = random.Random(seed)
    print(f"=== Core bình luận: {n} dòng ===\n\nPhân bố số từ (sau khi bỏ từ dừng):")
    for lo, hi, name in BUCKETS:
        c = sum(1 for k in t["n_tokens"] if lo <= k <= hi)
        print(f"  {name:8s} {c:7d}  {100 * c / n:5.1f}%")

    for is_reply, name in ((False, "bình luận gốc"), (True, "reply")):
        idx = [i for i, r in enumerate(t["is_reply"]) if bool(r) == is_reply]
        if idx:
            ok = sum(1 for i in idx if t["n_tokens"][i] >= 3)
            print(f"Đủ dài để đưa vào GSDMM ({name}): {100 * ok / len(idx):.1f}% của {len(idx)}")

    def show(title, idx, key):
        print(f"\n--- {title} (mẫu ngẫu nhiên {min(n_sample, len(idx))}/{len(idx)}) ---")
        for i in rng.sample(idx, min(n_sample, len(idx))):
            print("  •", repr(t[key][i])[:140])

    show("Rỗng sau khi làm sạch (hiện bản gốc)", [i for i, c in enumerate(t["text_clean"]) if c == ""], "text_raw")
    show("Có đúng 1-2 từ (hiện text_clean)", [i for i, k in enumerate(t["n_tokens"]) if k in (1, 2)], "text_clean")

    rep = Counter(c for c in t["text_clean"] if c)
    print("\n--- 15 bình luận lặp lại nhiều nhất (text_clean) ---")
    for text, c in rep.most_common(15):
        print(f"  {c:6d}  {text[:80]!r}")

    vfolder = config.DATA_DIR / "core" / "videos"
    if vfolder.exists():
        v = ds.dataset(str(vfolder), format="parquet").to_table(columns=["video_id", "title", "keyword"]).to_pydict()
        info = {vid: (ti, kw) for vid, ti, kw in zip(v["video_id"], v["title"], v["keyword"])}
        vc = ds.dataset(str(folder), format="parquet").to_table(columns=["video_id"]).to_pydict()["video_id"]
        print("\n--- 10 video có nhiều bình luận nhất ---")
        for vid, c in Counter(vc).most_common(10):
            ti, kw = info.get(vid, ("?", "?"))
            print(f"  {c:6d}  [{kw}] {str(ti)[:70]}")
            
    top = Counter(tok for toks in t["tokens"] for tok in (toks or []))
    print("\n--- 40 từ phổ biến nhất trong tokens ---")
    print("  " + ", ".join(f"{w}({c})" for w, c in top.most_common(40)))


if __name__ == "__main__":
    main()