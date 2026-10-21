"""Xem nhanh dữ liệu đã thu thập.

Chạy:  docker compose --profile debug run --rm airflow-cli python -m ytb.status
"""

from __future__ import annotations

from . import storage


def main() -> None:
    meta = storage.video_meta()
    cols = storage.read_columns("comments", ["comment_id", "is_reply"])
    first_seen: dict[str, bool] = {}
    for cid, is_reply in zip(cols["comment_id"], cols["is_reply"]):
        first_seen.setdefault(cid, bool(is_reply))
    n_reply = sum(first_seen.values())

    scans = storage.load_state("comment_scans.json", {})
    ok = sum(1 for s in scans.values() if s.get("status") == "ok")
    unavailable = sum(1 for s in scans.values() if s.get("status") == "unavailable")
    snap_days = sorted(set(storage.read_columns("video_snapshots", ["snapshot_date"])["snapshot_date"]))
    backfill = storage.load_state("backfill_state.json", {}).get("next_before", "(chưa bắt đầu)")

    print("=== Dữ liệu Staging ===")
    print(f"Video (không trùng):          {len(meta)}")
    print(f"  đã có siêu dữ liệu đầy đủ:  {sum(1 for m in meta.values() if m['complete'])}")
    print(f"Snapshot: {storage.count_rows('video_snapshots')} dòng, {len(snap_days)} ngày"
          + (f" ({snap_days[0]} đến {snap_days[-1]})" if snap_days else ""))
    print(f"Bình luận (không trùng):      {len(first_seen)}  (gốc {len(first_seen) - n_reply}, reply {n_reply})")
    print("=== Tiến độ quét bình luận ===")
    print(f"Đã quét xong:                 {ok}")
    print(f"Tắt bình luận/không truy cập: {unavailable}")
    print(f"Chưa quét:                    {max(0, len(meta) - ok - unavailable)}")
    print(f"Backfill video đang ở mốc:    {backfill}")


if __name__ == "__main__":
    main()