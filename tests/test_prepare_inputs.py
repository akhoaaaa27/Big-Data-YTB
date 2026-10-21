"""Trial outputs must be reversible and preserve different modeling populations."""
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ytb import config, filter_comments
from ytb.modeling.train_topics import train


def fake_language(text):
    if text.startswith("foreign"):
        return "fr", .99, .98, "other"
    if text.startswith("ambiguous"):
        return "vi", .4, .1, "uncertain"
    if text in ("❤️", "hay", ""):
        return "und", 0., 0., "short"
    return "vi", .99, .98, "vi"


@pytest.fixture
def trial_source():
    source = config.DATA_DIR / "core"
    (source / "videos").mkdir(parents=True)
    videos = [dict(video_id="a", title="Một bài hát Suno AI", default_audio_language="vi"),
              dict(video_id="b", title="Chương trình giải trí", default_audio_language="vi"),
              dict(video_id="c", title="Beautiful song Suno AI", default_audio_language="en"),
              dict(video_id="d", title="Hướng dẫn Suno AI", default_audio_language="vi")]
    pq.write_table(pa.Table.from_pylist(videos), source / "videos/part.parquet")
    rows = []
    for i in range(60):
        rows.append(dict(comment_id=f"c{i:03}", video_id="a", text_raw=f"bai hat rat hay {i}",
                         text_clean=f"bai hat rat hay {i}", tokens=["bai_hat", "hay", str(i)],
                         is_reply=False, is_modelable=True))
    for cid, text, vid, reply in [("emoji", "❤️", "a", False), ("short", "hay", "a", False),
                                  ("reply", "Cảm ơn bạn đã chia sẻ bài hát", "a", True),
                                  ("name", "Sơn Tùng hát bài này hay vãi", "a", False),
                                  ("foreign", "foreign chanson française très jolie", "a", False),
                                  ("uncertain", "ambiguous unaccented unknown", "a", False),
                                  ("spam", "Mua ngay https://spam.invalid", "a", False),
                                  ("ordinary", "bài hát rất hay", "b", False),
                                  ("foreign_video", "bài hát rất hay", "c", False),
                                  ("orphan", "bài hát rất hay", "missing", False),
                                  ("empty", "", "a", False)]:
        rows.append(dict(comment_id=cid, video_id=vid, text_raw=text, text_clean=text,
                         tokens=text.split(), is_reply=reply, is_modelable=True))
    (source / "comments").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), source / "comments/part.parquet")
    return source


def test_trial_membership_preservation_and_model_integration(trial_source):
    originals = {p: p.read_bytes() for p in trial_source.rglob("*.parquet")}
    result = filter_comments.prepare_inputs(detector=fake_language, cap_per_video=50)
    out = config.DATA_DIR / "filtered_comments"
    sentiment = pq.read_table(out / "sentiment/comments/part-00000.parquet").to_pylist()
    topics = pq.read_table(out / "topics/comments/part-00000.parquet").to_pylist()
    ids = {r["comment_id"] for r in sentiment}
    assert {"emoji", "short", "reply", "name"} <= ids
    assert not {"foreign", "uncertain", "spam", "ordinary", "foreign_video", "orphan", "empty"} & ids
    assert len(topics) == 50
    assert set(Counter(r["video_id"] for r in topics)) == {"a"}
    assert not {"emoji", "short", "reply"} & {r["comment_id"] for r in topics}
    assert result["source_sha256_before"] == result["source_sha256_after"]
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert not (config.DATA_DIR / "review").exists()
    assert not (config.DATA_DIR / "curated").exists()
    original_schema = pq.read_table(trial_source / "comments/part.parquet").schema
    actual = pq.read_table(out / "topics/comments/part-00000.parquet")
    for field in original_schema:
        assert actual.schema.field(field.name).type == field.type
    # The model must honor prepared VI detection even for unaccented text.
    trained = train(out / "topics/comments", config.DATA_DIR / "trial_model", k=2, iterations=1)
    assert trained["funnel"]["training"] == 50
    assert trained["language_filter_method"] == "prepared_language_status"


def test_trial_no_overwrite_and_partition_order_independence(trial_source):
    first = filter_comments.prepare_inputs(detector=fake_language, cap_per_video=5)
    with pytest.raises(ValueError, match="Không ghi đè"):
        filter_comments.prepare_inputs(detector=fake_language)
    file = trial_source / "comments/part.parquet"
    rows = pq.read_table(file).to_pylist()
    pq.write_table(pa.Table.from_pylist(rows[::-1]), file)
    second_dir = config.DATA_DIR / "second_trial"
    second = filter_comments.prepare_inputs(output=second_dir, detector=fake_language, cap_per_video=5)
    def ids(path):
        return set(pq.read_table(path / "topics/comments/part-00000.parquet")["comment_id"].to_pylist())
    assert ids(config.DATA_DIR / "filtered_comments") == ids(second_dir)
    assert first["topic_reasons"] == second["topic_reasons"]


@pytest.mark.parametrize("destination", ["core", "core/nested"])
def test_trial_rejects_source_overlap(trial_source, destination):
    with pytest.raises(ValueError, match="tách biệt"):
        filter_comments.prepare_inputs(output=config.DATA_DIR / destination, detector=fake_language)


def test_trial_write_failure_does_not_publish(trial_source, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("disk failure")
    monkeypatch.setattr(filter_comments.pq, "write_table", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        filter_comments.prepare_inputs(detector=fake_language)
    assert not (config.DATA_DIR / "filtered_comments").exists()
    assert not list(config.DATA_DIR.glob(".analysis-trial-*"))


def test_trial_empty_selection_still_has_schema(trial_source):
    def other(text):
        return "fr", .99, .98, "other"
    result = filter_comments.prepare_inputs(detector=other)
    assert result["topic_comments"] == result["sentiment_comments"] == 0
    assert "tokens" in pq.read_schema(config.DATA_DIR / "filtered_comments/topics/comments/part-00000.parquet").names


def test_real_language_detection_is_conservative():
    language = filter_comments.CommentLanguage()
    assert language("Bài hát này rất hay và giọng hát nghe rất tự nhiên")[3] == "vi"
    assert language("Cette chanson est très belle et je voudrais écouter la musique encore")[3] == "other"
    # An accent alone must not turn a French sentence into Vietnamese.
    assert language("hay quá ❤️")[3] == "short"
    assert language("❤️❤️❤️")[3] == "short"
    assert language("bai hat nay nghe rat hay va giong hat rat tu nhien")[3] == "uncertain"


@pytest.mark.parametrize("description,expected", [
    ("Chào mừng bạn!\n- AI Cover độc lạ\n- Nhạc dân tộc", "uncertain"),
    ("Thể hiện: AI Cover by Mộng Phiêu", "include"),
    ("Nhạc và ca sĩ: Suno AI", "include"),
    ("AI Cover • Hòa âm: Người hát | Suno AI", "include"),
    ("This song was generated by Suno AI", "include"),
])
def test_trial_requires_work_specific_description_credit(description, expected):
    assert filter_comments.classify_video({"title": "Một bài hát", "description": description})[0] == expected


@pytest.fixture
def mixed_input(tmp_path):
    date = datetime(2026, 10, 1)
    def row(cid, vid, text, source, reply=False):
        return dict(comment_id=cid, video_id=vid, thread_id=cid, parent_id=None,
                    is_reply=reply, author_channel_id="author", text_raw=text, text_clean=text,
                    tokens=["nhạc", "ai", "hay"], n_tokens=3, like_count=1,
                    total_reply_count=0, source=source, published_at=date, updated_at=date,
                    published_date=date.date(), is_modelable=True, processed_at=date)
    data = [row("r1", "a", "bai hat rat hay", "api"),
            row("r2", "a", "giong hat rat tu nhien", "legacy_csv"),
            row("r3", "a", "foreign musique et chanson", "api"),
            row("r4", "a", "❤️", "api"),
            row("r5", "b", "bài hát rất hay", "api"),
            row("r6", "a", "Cảm ơn bạn đã chia sẻ", "api", True),
            row("s1", "s", "bai hat rat hay", "synthetic"),
            row("s2", "s", "giong hat rat tu nhien", "synthetic"),
            row("s3", "s", "foreign cette chanson", "synthetic"),
            row("s4", "s", "hay", "synthetic")]
    source = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist(data), source)
    real = tmp_path / "real_videos"
    fake = tmp_path / "synthetic_videos"
    real.mkdir(); fake.mkdir()
    pq.write_table(pa.Table.from_pylist([
        dict(video_id="a", title="Một bài hát Suno AI", default_audio_language="vi"),
        dict(video_id="b", title="Chương trình giải trí", default_audio_language="vi")]), real / "part.parquet")
    pq.write_table(pa.Table.from_pylist([
        dict(video_id="s", title="GIẢ LẬP: Bài hát Suno AI", default_audio_language="vi")]), fake / "part.parquet")
    return source, real, fake


def test_comparison_separates_origins_keeps_18_columns_and_avoids_accent_filter(mixed_input, tmp_path):
    source, real, fake = mixed_input
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "comparison"
    result = filter_comments.compare_sources(source, real, fake, output, detector=fake_language,
                                             train_models=True, iterations=1)
    assert result["origins"]["real"]["topic_comments"] == 2
    assert result["origins"]["real"]["sentiment_comments"] == 4
    assert result["origins"]["synthetic"]["topic_comments"] == 2
    assert result["origins"]["synthetic"]["sentiment_comments"] == 3
    assert result["origins"]["synthetic"]["metadata_is_synthetic_fixture"] is True
    original = pq.read_schema(source)
    for origin in ("real", "synthetic"):
        assert Path(result["origins"][origin]["output"]) == (output / origin).resolve()
        assert Path(result["origins"][origin]["output"]).exists()
        for kind in ("topics", "sentiment"):
            table = pq.read_table(output / origin / kind / "comments/part-00000.parquet")
            assert len(table.column_names) == 18 and table.schema.equals(original, check_metadata=True)
            assert all((r["source"] == "synthetic") == (origin == "synthetic") for r in table.to_pylist())
        audit = pq.read_table(output / origin / "comments_audit.parquet")
        assert "text_raw" not in audit.column_names
        assert "comment_language_status" in audit.column_names
        model = result["models"][origin]
        assert model["funnel"]["training"] == 2  # unaccented VI survives upstream decision
        assert model["vi_heuristic"] is False
        assert Path(model["source"]).exists()
        assert (output / origin / "gsdmm/seed_42/topic_examples.json").exists()
    assert result["input_sha256_before"] == result["input_sha256_after"] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not (output / "_inputs").exists()


def test_comparison_no_overwrite_or_core_output(mixed_input, tmp_path):
    source, real, fake = mixed_input
    target = tmp_path / "already_exists"
    target.mkdir()
    with pytest.raises(FileExistsError):
        filter_comments.compare_sources(source, real, fake, target, detector=fake_language)
    with pytest.raises(ValueError, match="Core"):
        filter_comments.compare_sources(source, real, fake, config.DATA_DIR / "core/nested", detector=fake_language)


def test_comparison_rejects_unknown_provenance(mixed_input, tmp_path):
    source, real, fake = mixed_input
    table = pq.read_table(source)
    table = table.set_column(table.column_names.index("source"), "source", pa.array(["unknown"] * table.num_rows))
    pq.write_table(table, source)
    target = tmp_path / "comparison"
    with pytest.raises(ValueError, match="source phải"):
        filter_comments.compare_sources(source, real, fake, target, detector=fake_language)
    assert not target.exists()


def test_comparison_keeps_filter_reports_when_too_few_words_to_train(mixed_input, tmp_path):
    source, real, fake = mixed_input
    table = pq.read_table(source)
    pq.write_table(table.take(pa.array([0, 6])), source)  # one document per origin, all words occur once
    output = tmp_path / "tiny_comparison"
    result = filter_comments.compare_sources(source, real, fake, output, detector=fake_language,
                                             train_models=True, iterations=1)
    for origin in ("real", "synthetic"):
        assert result["models"][origin]["status"] == "skipped_empty_after_vocabulary_filter"
        assert (output / origin / "topics/comments/part-00000.parquet").exists()
    assert (output / "comparison.json").exists()
