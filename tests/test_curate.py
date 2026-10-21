import csv

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from ytb import config, curate, make_label_sample


def write_table(name, rows):
    path = config.DATA_DIR / 'core' / name
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path / 'part.parquet')


@pytest.fixture
def candidates():
    write_table('videos', [
        {'video_id': 'a', 'title': 'Một bài hát - Suno AI', 'keyword': 'nhạc AI'},
        {'video_id': 'b', 'title': 'Ai đưa em về', 'keyword': 'nhạc AI'},
        {'video_id': 'c', 'title': 'Hướng dẫn làm nhạc Suno AI', 'keyword': 'AI cover'},
    ])
    write_table('comments', [
        {'comment_id': vid, 'video_id': vid, 'tokens': ['hay'], 'is_reply': False}
        for vid in ('a', 'b', 'c', 'orphan')
    ])
    write_table('video_snapshots', [{'video_id': vid, 'view_count': 100} for vid in ('a', 'b', 'c', 'orphan')])


@pytest.mark.parametrize('video,expected', [
    ({'title': 'Ai đưa em về', 'keyword': 'nhạc AI', 'contains_synthetic_media': True}, 'uncertain'),
    ({'title': 'Ai là người đang hát???'}, 'uncertain'),
    ({'title': 'Sơn Tùng hát live', 'channel_title': 'Nhạc AI'}, 'uncertain'),
    ({'title': '5 BƯỚC TẠO NHẠC SUNO AI'}, 'exclude'),
    ({'title': 'a New RTS Game with deep AI cover mechanics | Defilade'}, 'exclude'),
    ({'title': 'Tóm tắt phim tổng tài'}, 'exclude'),
    ({'title': 'Test Nhạc AI Bass Cực Trầm Với Bộ Âm Thanh'}, 'exclude'),
    ({'title': 'Bài hát', 'description': 'Nhạc và ca sĩ: Suno AI'}, 'include'),
    ({'title': 'Bài hát', 'description': 'Generated AI images', 'contains_synthetic_media': True}, 'uncertain'),
    ({'title': 'AI Art Music Video', 'tags': ['AI music']}, 'uncertain'),
    ({'title': 'Lời yêu (A.I hát)'}, 'include'),
    ({'title': 'HOA SỨ - NHẠC COVER AI'}, 'include'),
    ({'title': 'Một bản nhạc #aimusic #sunoai'}, 'uncertain'),
    ({'title': 'AI Cover', 'description': 'This is not AI; recorded by a real singer'}, 'uncertain'),
    ({'title': 'Bài hát thường', 'tags': ['AI cover', 'Suno AI']}, 'uncertain'),
    ({'title': 'Trend Nhạc Ai Đưa Em Về | Bạn thích ai hơn'}, 'uncertain'),
    ({'title': 'Nhạc Ai Đưa Em Về - AI Cover'}, 'include'),
    ({'title': 'Beautiful Sunday', 'channel_title': 'Leo Music AI', 'description': 'Beautiful Sunday | Leo Music AI\nFollow Leo Music AI'}, 'uncertain'),
    ({'title': 'Forget Suno. New BEST local AI music generator is here!'}, 'exclude'),
    ({'title': 'Suno AI đã bị cấm! Chuyện gì tiếp theo?'}, 'exclude'),
    ({'title': 'Rác AI Trên YouTube Đã Tệ Đến Mức Này (AI Slop)', 'description': 'YouTube tự gắn bản remix AI'}, 'exclude'),
    ({'title': 'Bài hát', 'description': 'Kênh chuyên chia sẻ những bản nhạc AI'}, 'uncertain'),
    ({'title': 'Bản tin AI cover', 'category_id': '25'}, 'uncertain'),
    ({'title': 'Tạo MV ca nhạc Bằng AI Miễn Phí - Quá Đơn Giản'}, 'exclude'),
    ({'title': 'Cây Cầu Tình Ái Cover'}, 'uncertain'),
    ({'title': 'BẾN VẮNG CHỜ AI Remix'}, 'uncertain'),
    ({'title': 'Vì Ai - AI Cover'}, 'include'),
    ({'title': '🔥🔥 #aicover #dance #dancer'}, 'uncertain'),
])
def test_classification(video, expected):
    assert curate.classify(video)[0] == expected


def test_auto_filters_all_tables_without_manual_labels_and_keeps_core(candidates):
    originals = {p: p.read_bytes() for p in (config.DATA_DIR / 'core').rglob('*.parquet')}
    # File theo cơ chế cũ không được biến thành đầu vào bắt buộc hay ghi đè thuật toán.
    curate.review_path().parent.mkdir()
    curate.review_path().write_text('video_id,decision\na,exclude\nb,include\n')
    result = curate.run_auto()
    assert result['decisions'] == {'include': 1, 'exclude': 1, 'uncertain': 1}
    assert result['comments'] == {'before': 4, 'kept': 1, 'removed': 3}
    for name in ('videos', 'comments', 'video_snapshots'):
        source = ds.dataset(config.DATA_DIR / 'core' / name, format='parquet')
        out = ds.dataset(config.DATA_DIR / 'curated' / name, format='parquet')
        assert out.schema.equals(source.schema)
        assert out.to_table()['video_id'].to_pylist() == ['a']
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert curate.run_auto()['comments'] == result['comments']
    assert not (config.DATA_DIR / 'curated.old').exists()
    with (config.DATA_DIR / 'curated' / 'uncertain.csv').open(encoding='utf-8-sig') as f:
        assert [r['video_id'] for r in csv.DictReader(f)] == ['b']


def test_export_is_automatic_not_a_form_to_fill(candidates):
    assert curate.export_review()['decisions'] == {'include': 1, 'exclude': 1, 'uncertain': 1}
    with curate.review_path().open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert all(r['decision'] in ('include', 'exclude', 'uncertain') for r in rows)
    assert not (config.DATA_DIR / 'curated').exists()


def test_no_matches_publishes_empty_tables_instead_of_leaking_old_rows(candidates):
    curate.run_auto()
    path = config.DATA_DIR / 'core' / 'videos' / 'part.parquet'
    pq.write_table(pa.Table.from_pylist([{'video_id': 'a', 'title': 'Không đủ thông tin'}]), path)
    result = curate.run_auto()
    assert result['decisions']['include'] == 0
    assert ds.dataset(config.DATA_DIR / 'curated' / 'comments', format='parquet').count_rows() == 0


def test_failed_run_preserves_previous_curated(candidates, monkeypatch):
    curate.run_auto()
    before = (config.DATA_DIR / 'curated' / 'comments' / 'part-00000.parquet').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(curate.pq, 'ParquetWriter', fail)
    with pytest.raises(OSError, match='disk full'):
        curate.run_auto()
    assert (config.DATA_DIR / 'curated' / 'comments' / 'part-00000.parquet').read_bytes() == before
    assert not list(config.DATA_DIR.glob('curated-building-*'))


def test_labeling_refuses_unfiltered_core(candidates):
    with pytest.raises(SystemExit, match='Chưa có dữ liệu lọc tự động'):
        make_label_sample.load_candidates()
