"""Kiểm tra luồng dữ liệu; predictor giả trong test không phải kết quả VisoBERT."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jobs import assign_aspects as aspects
from jobs import predict_sentiment as sentiment

ROOT = Path(__file__).resolve().parents[1]


def make_silver(destination):
    destination.mkdir()
    examples = [
        ("Giai điệu nghe hay quá", ["giai_điệu", "hay"]),
        ("Bản phối này chưa hay", ["bản_phối", "chưa", "hay"]),
        ("Có bản karaoke không ad", ["bản", "karaoke", "không"]),
        ("Âm thanh có hơi rè", ["âm_thanh", "hơi", "rè"]),
    ]
    rows = []
    for index in range(24):
        text, tokens = examples[index % len(examples)]
        rows.append({"comment_id": f"fixture-{index:03}", "video_id": f"fixture-v{index % 3}",
                     "author": None, "text": f"{text}, phiên bản {index}", "likes": index,
                     "published": "2026-10-20", "tokens": tokens})
    rows.extend([
        {**rows[0], "text": "Bản ghi trùng ID không được nhân đôi"},
        {**rows[0], "comment_id": "fixture-copy"},
        {**rows[0], "comment_id": "fixture-empty", "text": "  "},
        {**rows[0], "comment_id": "fixture-rare", "text": "Hai từ chỉ gặp một lần",
         "tokens": ["hiếm_một", "hiếm_hai"]},
        {**rows[0], "comment_id": "fixture-one", "text": "Một token đơn lẻ",
         "tokens": ["hay"]},
    ])
    table = pa.Table.from_pylist(rows)
    pq.write_table(table.slice(0, 13), destination / "part-00000.parquet")
    pq.write_table(table.slice(13), destination / "part-00001.parquet")
    (destination / "_SUCCESS").touch()
    (destination / ".part-00000.parquet.crc").write_bytes(b"fixture checksum placeholder")
    return table


class FixturePredictor:
    metadata = {"model": "TEST_FIXTURE_ONLY_NOT_REAL_PREDICTIONS"}

    def predict(self, texts):
        return [({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, False)
                for _ in texts]


def test_gsdmm_single_aspect_sentiment_pipeline(tmp_path):
    silver = tmp_path / "silver"
    make_silver(silver)
    original_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in silver.glob("*.parquet")}
    gold = tmp_path / "gold"
    command = [sys.executable, str(ROOT / "jobs/gsdmm_train.py"), "42",
               "--input", str(silver), "--output-dir", str(gold), "--iterations", "3"]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((gold / "training_summary.json").read_text(encoding="utf-8"))
    assert report["input_rows"] == 29
    assert len(report["input_files"]) == 2
    assert report["training_rows"] == 24
    assert report["min_tokens"] is None
    clustered = pq.read_table(gold / "comments_clustered.parquet")
    assert len(set(clustered["comment_id"].to_pylist())) == len(clustered) == 24
    # Bình luận có đúng 2 token vẫn được dùng: không tái thêm gate >= 3.
    assert "fixture-000" in clustered["comment_id"].to_pylist()

    mapping_path = gold / "new_mapping.json"
    aspects.prepare_mapping(gold / "cluster_top_words.csv", mapping_path)
    config = json.loads(mapping_path.read_text(encoding="utf-8"))
    first_cluster = str(clustered["cluster"][0].as_py())
    config["cluster_aspects"][first_cluster] = "Giai điệu"
    mapping_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    result_dir = gold / "aspects"
    label_report = aspects.label_comments(gold / "comments_clustered.parquet",
                                         gold / "cluster_top_words.csv", result_dir, mapping_path)
    labeled = pq.read_table(result_dir / "comments_with_aspects.parquet")
    assert labeled.select(clustered.column_names).equals(clustered)
    assert labeled.column_names.count("aspect") == 1
    assert set(labeled["aspect"].to_pylist()) <= {"Giai điệu", "Khác"}
    assert len(labeled) == len(clustered) == label_report["rows"]
    assert all(isinstance(value, str) and value for value in labeled["aspect"].to_pylist())

    sentiment_dir = gold / "sentiment"
    sentiment_report = sentiment.predict(result_dir / "comments_with_aspects.parquet",
                                         sentiment_dir, predictor=FixturePredictor(), batch_size=7)
    final = pq.read_table(sentiment_dir / "comments_sentiment.parquet")
    assert final.select(labeled.column_names).equals(labeled)
    assert sentiment_report["output_rows"] == 24
    with (sentiment_dir / "aspect_sentiment_summary.csv").open(encoding="utf-8-sig") as stream:
        assert sum(int(row["comments"]) for row in csv.DictReader(stream)) == 24

    # Các lần chạy lặp không được xóa hoặc ghi đè dữ liệu đã có.
    clustered_before = hashlib.sha256((gold / "comments_clustered.parquet").read_bytes()).hexdigest()
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0 and "Đã có kết quả" in repeat.stderr
    assert hashlib.sha256((gold / "comments_clustered.parquet").read_bytes()).hexdigest() == clustered_before
    with pytest.raises(FileExistsError):
        aspects.label_comments(gold / "comments_clustered.parquet", gold / "cluster_top_words.csv",
                               result_dir, mapping_path)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in silver.glob("*.parquet")} == original_hashes


def test_reference_mapping_matches_users_cluster_counts():
    rows, signature = aspects.read_words(ROOT / "config/cluster_top_words_seed42.csv")
    mapping = aspects.load_mapping(ROOT / "config/cluster_aspects_seed42.json",
                                   signature, [row["cluster"] for row in rows])
    counts = Counter()
    for row in rows:
        counts[mapping[row["cluster"]]] += row["so_binh_luan"]
    assert sum(counts.values()) == 34923
    assert counts["Khác"] == 12953
    assert counts["Minh bạch và ghi công"] == 4184
    assert counts["Lời bài hát và cảm xúc"] == 991


def test_mapping_from_other_training_is_rejected():
    with pytest.raises(ValueError, match="Bảng từ khóa khác"):
        aspects.load_mapping(ROOT / "config/cluster_aspects_seed42.json", "wrong-run", [0])


@pytest.mark.parametrize("invalid", [["Giọng hát", "Giai điệu"], "", None])
def test_multi_aspect_or_empty_mapping_is_rejected(tmp_path, invalid):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"cluster_words_signature": "same-run",
                                "cluster_aspects": {"0": invalid}}), encoding="utf-8")
    with pytest.raises(ValueError, match="đúng một tên"):
        aspects.load_mapping(path, "same-run", [0])
