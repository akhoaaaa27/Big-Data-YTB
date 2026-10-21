import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ytb import config, make_label_sample as ls


@pytest.fixture
def core():
    d = config.DATA_DIR / "core"
    (d / "comments").mkdir(parents=True)
    (d / "videos").mkdir(parents=True)
    rows = []
    for v in range(40):                                   # 40 video, mỗi video 10 bình luận
        for k in range(10):
            rows.append({"comment_id": f"c{v}_{k}", "video_id": f"v{v}", "is_reply": k % 5 == 0,
                         "text_raw": f"Bình luận số {k} <br>của video {v} &amp; ❤" if k else "",
                         "text_clean": f"binh luan {k}" if k else "", "is_modelable": True, "source": "api"})
    pq.write_table(pa.Table.from_pylist(rows), d / "comments" / "part-0.parquet")
    pq.write_table(pa.Table.from_pylist([{"video_id": f"v{v}", "title": f"Video {v}", "keyword": "nhạc ai"} for v in range(40)]),
                   d / "videos" / "part-0.parquet")


def read(name):
    with (ls.label_dir() / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_chia_mau_chung_rieng_khong_trung_va_trai_deu_video(core):
    s = ls.run(["A", "B", "C", "D"], shared=10, per_person=20, cap=3, seed=1)
    assert s["_unique_comments"] == 10 + 4 * 20 and all(s[n] == 30 for n in "ABCD")
    sheets = {n: read(f"phieu_{n}.csv") for n in "ABCD"}
    ids = {n: {r["comment_id"] for r in sheets[n]} for n in sheets}
    shared = set.intersection(*ids.values())
    assert len(shared) == 10                                              # đúng 10 bình luận chung
    own = [ids[n] - shared for n in "ABCD"]
    assert all(len(o) == 20 for o in own) and len(set().union(*own)) == 80   # phần riêng không trùng nhau
    master = read("sample_master.csv")
    per_video = {}
    for r in {r["comment_id"]: r for r in master}.values():
        per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
    assert max(per_video.values()) <= 3                                   # mỗi video tối đa 3
    assert all("".join(r["binh_luan"].split()) for r in master)           # không có bình luận rỗng
    assert "<br" not in sheets["A"][0]["binh_luan"] and "&amp;" not in sheets["A"][0]["binh_luan"]
    assert set(sheets["A"][0]) == {"stt", "comment_id", "video_title", "binh_luan", "cam_xuc", "chu_de_ghi_chu", "ghi_chu"}


def test_cung_seed_ra_cung_mau_va_khong_ghi_de(core):
    ls.run(["A", "B"], 5, 10, 3, seed=7)
    first = read("phieu_A.csv")
    with pytest.raises(SystemExit):
        ls.run(["A", "B"], 5, 10, 3, seed=7)                               # đã có mẫu: từ chối
    ls.run(["A", "B"], 5, 10, 3, seed=7, force=True)
    assert read("phieu_A.csv") == first                                    # cùng seed, cùng kết quả


def test_bao_loi_khi_khong_du_binh_luan(core):
    with pytest.raises(SystemExit):
        ls.run(["A", "B", "C", "D"], 50, 62, cap=1, seed=1)                # 40 video x cap 1 = 40 < 298