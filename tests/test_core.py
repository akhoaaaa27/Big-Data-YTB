import pytest

from ytb import config, storage, text_clean
from ytb.spark_core import build_spark, core_dir, run_core


@pytest.fixture(scope="module")
def spark():
    s = build_spark("test_core")
    yield s
    s.stop()


def cmt(cid, text, updated="2026-10-01T00:00:00Z", fetched="2026-10-02T00:00:00Z", **kw):
    row = {"comment_id": cid, "video_id": "v1", "thread_id": cid, "parent_id": None, "is_reply": False,
           "author_display_name": "Ai Đó", "author_channel_id": "UCx", "text": text, "like_count": 1,
           "published_at": "2026-09-30T10:00:00Z", "updated_at": updated, "total_reply_count": 0,
           "source": "api", "fetched_at": fetched}
    row.update(kw)
    return row


# ---------------- làm sạch (không cần Spark) ----------------
def test_clean_text_xu_ly_html_link_mention_moc_thoi_gian_va_chu_lap():
    raw = "Hay quáaaaa 3:45 <a href=\"x\">xem</a> @Ban_A https://youtu.be/abc &amp; nghe lại nha 😍😍"
    out = text_clean.clean_text(raw)
    assert out == "hay quá xem & nghe lại nha 😍😍"      # giữ emoji cho mô hình cảm xúc


def test_clean_text_rong_hoac_none():
    assert text_clean.clean_text(None) == "" and text_clean.clean_text("   ") == ""


def test_tokenize_bo_emoji_dau_cau_tu_dung_va_so():
    toks = text_clean.tokenize(text_clean.clean_text("Giọng hát này nghe rất hay, như người thật 😍 100"))
    assert "😍" not in " ".join(toks) and "100" not in toks
    assert "này" not in toks and "rất" not in toks
    assert "giọng" in " ".join(toks)


# ---------------- Spark ----------------
def test_core_khu_trung_giu_ban_moi_nhat_va_lam_sach(spark):
    storage.write_rows("comments", [cmt("c1", "Bản cũ")], ingest_date="2026-10-02")
    storage.write_rows("comments", [
        cmt("c1", "Bản MỚI hay quáaaa nghe cảm động lắm", updated="2026-10-05T00:00:00Z", fetched="2026-10-06T00:00:00Z"),
        cmt("c2", "👍", parent_id="c1", is_reply=True, thread_id="c1"),
    ], ingest_date="2026-10-06")
    s = run_core(spark)["comments"]
    assert s["staging_rows"] == 3 and s["core_rows"] == 2 and s["duplicates_removed"] == 1
    rows = {r["comment_id"]: r for r in spark.read.parquet(str(core_dir() / "comments")).collect()}
    assert rows["c1"]["text_raw"].startswith("Bản MỚI")
    assert rows["c1"]["text_clean"] == "bản mới hay quá nghe cảm động lắm"
    assert rows["c1"]["is_modelable"] is True
    assert rows["c2"]["n_tokens"] == 0 and rows["c2"]["is_modelable"] is False
    assert s["empty_after_cleaning"] == 0 and s["modelable"] == 1
    assert "author_display_name" not in rows["c1"].asDict()          # không giữ tên hiển thị ở Core
    assert rows["c1"]["published_date"].isoformat() == "2026-09-30"


def test_core_chay_lai_cung_ket_qua_va_khong_de_thu_muc_do_dang(spark):
    storage.write_rows("comments", [cmt("c1", "Bài này nghe hay thật sự")], ingest_date="2026-10-02")
    a = run_core(spark)
    b = run_core(spark)
    assert a["comments"]["core_rows"] == b["comments"]["core_rows"] == 1
    assert not list(core_dir().glob("*.old")) and not (core_dir() / "_building" / "comments").exists()


def test_core_video_uu_tien_ban_api_va_snapshot_moi_ngay_mot_dong(spark):
    storage.write_rows("videos", [
        {"video_id": "v1", "title": "cũ", "source": "legacy_csv", "fetched_at": "2026-10-09T00:00:00Z",
         "published_at": "2026-09-01T00:00:00Z"},
        {"video_id": "v1", "title": "đầy đủ", "channel_id": "UC1", "source": "api", "fetched_at": "2026-10-01T00:00:00Z",
         "published_at": "2026-09-01T00:00:00Z"},
    ], ingest_date="2026-10-10")
    storage.write_rows("video_snapshots", [
        {"video_id": "v1", "snapshot_date": "2026-10-10", "snapshot_at": "2026-10-10T01:00:00Z", "view_count": 5},
        {"video_id": "v1", "snapshot_date": "2026-10-10", "snapshot_at": "2026-10-10T09:00:00Z", "view_count": 9},
    ], ingest_date="2026-10-10")
    s = run_core(spark)
    v = spark.read.parquet(str(core_dir() / "videos")).collect()
    assert len(v) == 1 and v[0]["title"] == "đầy đủ" and v[0]["channel_id"] == "UC1"
    sn = spark.read.parquet(str(core_dir() / "video_snapshots")).collect()
    assert len(sn) == 1 and sn[0]["view_count"] == 9
    assert "comments" not in s


def test_core_chua_co_du_lieu_staging_thi_khong_loi(spark):
    assert run_core(spark) == {}


def test_tokenize_khong_bo_mat_tu_ai_va_hay():
    words = {w for tok in text_clean.tokenize(text_clean.clean_text("Nhạc AI hay quá")) for w in tok.split("_")}
    assert "ai" in words and "hay" in words


def test_tokenize_bo_tu_dung_tieng_anh_nhung_giu_tu_viet_khong_dau():
    words = {w for tok in text_clean.tokenize(text_clean.clean_text("bài này do AI hát, the best song, to quá, bé hát")) for w in tok.split("_")}
    assert "do" in words and "to" in words and "best" in words
    assert "the" not in words