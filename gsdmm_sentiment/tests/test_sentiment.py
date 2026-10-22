import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jobs import predict_sentiment as module


def test_label_order_comes_from_config():
    assert module.model_labels({0: 'NEG', 1: 'POS', 2: 'NEU'}) == {0: 'negative', 1: 'positive', 2: 'neutral'}
    assert module.model_labels({'0': 'positive', '1': 'negative', '2': 'neutral'}) == {0: 'positive', 1: 'negative', 2: 'neutral'}


def test_ambiguous_labels_not_guessed():
    with pytest.raises(ValueError, match='Không rõ nhãn'):
        module.model_labels({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})
    assert module.model_labels({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'},
                               {'0': 'neutral', '1': 'positive', '2': 'negative'})[0] == 'neutral'
    with pytest.raises(ValueError, match='khác nhau'):
        module.model_labels({0: 'NEG', 1: 'NEG', 2: 'POS'})


def test_keep_negations_emoji_case():
    assert module.clean_for_sentiment('<b>AI KHÔNG hay 😕</b> https://x.test/a') == 'AI KHÔNG hay 😕'
    assert module.clean_for_sentiment(None) == ''


class FakeModel:
    metadata = {'model': 'TEST_FIXTURE_ONLY_NOT_REAL_PREDICTIONS'}

    def __init__(self):
        self.seen = []

    def predict(self, texts):
        self.seen.extend(texts)
        rows = []
        for text in texts:
            if '?' in text:
                rows.append(({'negative': .1, 'neutral': .7, 'positive': .2}, True))
            elif 'không' in text:
                rows.append(({'negative': .8, 'neutral': .1, 'positive': .1}, False))
            else:
                rows.append(({'negative': .05, 'neutral': .05, 'positive': .9}, False))
        return rows


def fixture_table():
    return pa.table({
        'comment_id': ['a', 'b', 'c', 'd'], 'video_id': ['v1', 'v2', 'v3', 'v4'],
        'aspect': ['Giọng hát', 'Khác', 'Phối khí và nhạc cụ', 'Karaoke và tông giọng'],
        'cluster': [1, 0, 9, 8],
        'text': ['Giọng AI hay 😍', 'không hay 😕', '   ', 'Có karaoke không?'],
        'tokens': [['giọng', 'hay'], ['hay'], [], ['karaoke']],
    })


def test_end_to_end_io_preserves_aspects_and_empty_rows(tmp_path):
    source = tmp_path / 'input.parquet'
    pq.write_table(fixture_table(), source)
    original = pq.read_table(source)
    fake = FakeModel()
    report = module.predict(source, tmp_path / 'output', batch_size=2, predictor=fake)
    result = pq.read_table(tmp_path / 'output/comments_sentiment.parquet')
    assert result.select(original.column_names).equals(original, check_metadata=True)
    assert result['sentiment'].to_pylist() == ['Tích cực', 'Tiêu cực', None, 'Trung lập']
    assert result['sentiment_truncated'].to_pylist() == [False, False, False, True]
    assert result['sentiment_score'].to_pylist()[2] is None
    assert fake.seen == ['Giọng AI hay 😍', 'không hay 😕', 'Có karaoke không?']
    assert report['predicted'] == 3 and report['empty_text'] == 1
    assert report['input_rows'] == report['output_rows'] == 4
    assert report['input_sha256_before'] == report['input_sha256_after']
    assert report['sentiment_counts'] == {'Tích cực': 1, 'Tiêu cực': 1, 'Trung lập': 1}
    with (tmp_path / 'output/comments_sentiment.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and rows[1]['aspect'] == 'Khác' and rows[1]['sentiment'] == 'Tiêu cực'
    assert json.loads(rows[1]['tokens']) == ['hay']
    with (tmp_path / 'output/aspect_sentiment_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        stats = list(csv.DictReader(stream))
    assert sum(int(row['comments']) for row in stats) == 4
    with pytest.raises(FileExistsError):
        module.predict(source, tmp_path / 'output', predictor=fake)


def test_trial_limit_is_explicit(tmp_path):
    source = tmp_path / 'input.parquet'
    pq.write_table(fixture_table(), source)
    report = module.predict(source, tmp_path / 'trial', limit=2, predictor=FakeModel())
    assert report['input_rows'] == 4 and report['output_rows'] == 2 and report['limit'] == 2
    assert len(pq.read_table(tmp_path / 'trial/comments_sentiment.parquet')) == 2


@pytest.mark.parametrize('probabilities', [
    {'positive': 1.0},
    {'negative': .1, 'neutral': .1, 'positive': .1},
    {'negative': float('nan'), 'neutral': .1, 'positive': .9},
])
def test_bad_predictions_not_published(tmp_path, probabilities):
    source = tmp_path / 'input.parquet'
    pq.write_table(fixture_table().slice(0, 1), source)
    fake = FakeModel()
    fake.predict = lambda texts: [(probabilities, False)]
    with pytest.raises(ValueError):
        module.predict(source, tmp_path / 'bad', predictor=fake)
    assert not (tmp_path / 'bad').exists()
