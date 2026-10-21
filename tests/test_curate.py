import csv

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from ytb import config, curate, make_label_sample


def write_table(name, rows):
    path = config.DATA_DIR / "core" / name
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "part.parquet")


@pytest.fixture
def candidates():
    write_table("videos", [
        {"video_id": "a", "title": "Một bài hát - Suno AI", "keyword": "nhạc AI"},
        {"video_id": "b", "title": "Ai đưa em về", "keyword": "nhạc AI"},
        {"video_id": "c", "title": "AI Cover", "keyword": "AI cover"},
    ])
    write_table("comments", [
        {"comment_id": vid, "video_id": vid, "tokens": ["hay"], "is_reply": False}
        for vid in ("a", "b", "c", "orphan")
    ])
    write_table("video_snapshots", [{"video_id": vid, "view_count": 100} for vid in ("a", "b", "c", "orphan")])


def decisions(**values):
    path = curate.review_path()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["decision"] = values.get(row["video_id"], row["decision"])
        row["note"] = "Đã kiểm tra"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curate.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_suggestions_do_not_trust_search_keyword_or_synthetic_flag():
    assert curate.suggest({"title": "Ai đưa em về", "keyword": "nhạc AI", "contains_synthetic_media": True})[0] == "review"
    assert curate.suggest({"title": "5 BƯỚC TẠO NHẠC SUNO AI"})[0] == "likely_other"
    assert curate.suggest({"title": "Bài hát", "description": "Nhạc và ca sĩ: Suno AI"})[0] == "likely_ai_music"


def test_export_never_auto_approves_and_preserves_manual_decisions(candidates):
    assert curate.export_review()["pending"] == 3
    assert not any(r["decision"] for r in curate.read_decisions().values())
    decisions(a="include", b="exclude")
    assert curate.export_review()["pending"] == 1
    assert curate.read_decisions()["a"]["decision"] == "include"
    assert curate.read_decisions()["a"]["note"] == "Đã kiểm tra"


def test_apply_filters_all_tables_and_keeps_core_untouched(candidates):
    originals = {p: p.read_bytes() for p in (config.DATA_DIR / "core").rglob("*.parquet")}
    curate.export_review()
    decisions(a="include", b="exclude")
    result = curate.apply_review()
    assert result["comments"] == {"before": 4, "kept": 1, "removed": 3}
    assert result["pending_videos"] == 1
    for table in ("videos", "comments", "video_snapshots"):
        source = ds.dataset(config.DATA_DIR / "core" / table, format="parquet")
        out = ds.dataset(config.DATA_DIR / "curated" / table, format="parquet")
        assert out.schema.equals(source.schema)
        assert out.to_table()["video_id"].to_pylist() == ["a"]
    assert all(p.read_bytes() == content for p, content in originals.items())
    decisions(a="exclude", c="include")
    curate.apply_review()
    assert ds.dataset(config.DATA_DIR / "curated" / "comments", format="parquet").to_table()["video_id"].to_pylist() == ["c"]
    assert not (config.DATA_DIR / "curated.old").exists()


def test_no_approval_or_invalid_decision_never_publishes(candidates):
    curate.export_review()
    with pytest.raises(ValueError, match="Chưa có video"):
        curate.apply_review()
    assert not (config.DATA_DIR / "curated").exists()
    decisions(a="yes")
    with pytest.raises(ValueError, match="không hợp lệ"):
        curate.apply_review()


def test_failed_apply_preserves_previous_curated(candidates, monkeypatch):
    curate.export_review()
    decisions(a="include")
    curate.apply_review()
    before = (config.DATA_DIR / "curated" / "comments" / "part-00000.parquet").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(curate.pq, "ParquetWriter", fail)
    with pytest.raises(OSError, match="disk full"):
        curate.apply_review()
    assert (config.DATA_DIR / "curated" / "comments" / "part-00000.parquet").read_bytes() == before
    assert not list(config.DATA_DIR.glob("curated-building-*"))


def test_labeling_refuses_unreviewed_core(candidates):
    with pytest.raises(SystemExit, match="Chưa có dữ liệu đã duyệt"):
        make_label_sample.load_candidates()
