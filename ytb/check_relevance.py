"""Đếm video đạt/bị loại bởi bộ lọc nhạc AI và in mẫu ngẫu nhiên để nhóm kiểm bằng mắt.

Chạy:  docker compose --profile debug run --rm airflow-cli python -m ytb.check_relevance
"""

from __future__ import annotations

import random

from . import storage


def main(n: int = 15, seed: int = 1) -> None:
    c = storage.read_columns("videos", ["video_id", "title", "keyword", "is_ai_music"])
    rows: dict[str, dict] = {}
    for vid, title, kw, ok in zip(c["video_id"], c["title"], c["keyword"], c["is_ai_music"]):
        r = rows.setdefault(vid, {"title": title, "keyword": kw, "ok": False})
        r["ok"] = r["ok"] or bool(ok)
    good = [r for r in rows.values() if r["ok"]]
    bad = [r for r in rows.values() if not r["ok"]]
    print(f"Video: {len(rows)}  đạt: {len(good)}  bị loại: {len(bad)}")
    rng = random.Random(seed)
    for name, group in (("ĐẠT", good), ("BỊ LOẠI", bad)):
        print(f"\n--- {name} (mẫu {min(n, len(group))}/{len(group)}) ---")
        for r in rng.sample(group, min(n, len(group))):
            print(f"  [{r['keyword']}] {str(r['title'])[:90]}")


if __name__ == "__main__":
    main()