"""Replace simulation-prefixed IDs with format-compatible fictional IDs.

Only identifiers in synthetic rows change. Existing provenance is preserved.
Run from the repository root; requires pyarrow, not Spark or the YouTube API.
"""

import argparse
import base64
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ID_COLUMNS = ("comment_id", "video_id", "thread_id", "parent_id", "author_channel_id")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_hash(folder):
    files = sorted(Path(folder).rglob("*.parquet"))
    return {str(p.relative_to(folder)): file_hash(p) for p in files}


def rewrite(comments, videos, real_core, output):
    comments, videos, real_core, output = map(Path, (comments, videos, real_core, output))
    output = output.resolve()
    for source in (comments, videos, real_core):
        source = source.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from all input paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    before = {"comments": file_hash(comments), "videos": metadata_hash(videos)}
    table = pq.read_table(comments)
    if len(table.schema) != 18:
        raise ValueError("Expected the original 18-column Core comment schema")
    rows = table.to_pylist()
    synthetic = [row for row in rows if row["source"] == "synthetic"]
    if not synthetic:
        raise ValueError("No synthetic rows found")
    video_table = ds.dataset(videos, format="parquet").to_table()
    video_rows = video_table.to_pylist()
    if len({r['video_id'] for r in video_rows}) != len(video_rows):
        raise ValueError("Video metadata IDs must be unique")
    if set(r['video_id'] for r in synthetic) - set(r['video_id'] for r in video_rows):
        raise ValueError("Some synthetic video IDs have no metadata")
    real_comments = ds.dataset(real_core / "comments", format="parquet").to_table(columns=list(ID_COLUMNS)).to_pylist()
    real_videos = ds.dataset(real_core / "videos", format="parquet").to_table(columns=["video_id", "channel_id"]).to_pylist()
    real_rows = real_comments + [r for r in rows if r['source'] != 'synthetic']
    reserved = {
        "comment": {r[k] for r in real_rows for k in ('comment_id', 'thread_id', 'parent_id') if r[k]},
        "video": {r['video_id'] for r in real_rows + real_videos if r['video_id']},
        "channel": {r['author_channel_id'] for r in real_rows if r['author_channel_id']} |
                   {r['channel_id'] for r in real_videos if r['channel_id']},
    }
    namespaces = {
        "comment": {r[k] for r in synthetic for k in ('comment_id', 'thread_id', 'parent_id') if r[k]},
        "video": {r['video_id'] for r in video_rows},
        "channel": {r['author_channel_id'] for r in synthetic if r['author_channel_id']} |
                   {r['channel_id'] for r in video_rows if r['channel_id']},
    }
    if any(not value.startswith('SYN_') for values in namespaces.values() for value in values):
        raise ValueError("Expected SYN_ simulation IDs; refusing to rewrite unfamiliar identifiers")
    mappings = {name: {} for name in namespaces}
    reply_parents = {r['comment_id']: r['parent_id'] for r in synthetic if r['is_reply']}
    if any(not value for value in reply_parents.values()):
        raise ValueError("Synthetic reply has no parent_id")

    def allocate(namespace, old, prefix='', length=11):
        attempt = 0
        while True:
            payload = f'ytb-simulation-id-v1:{namespace}:{old}:{attempt}'.encode()
            token = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
            value = prefix + token[:length]
            if 'syn' not in value.casefold() and value not in reserved[namespace]:
                reserved[namespace].add(value)
                return value
            attempt += 1

    active = set()

    def comment_id(old):
        if old in mappings['comment']:
            return mappings['comment'][old]
        if old in active:
            raise ValueError('Cycle in comment parent links')
        active.add(old)
        if old in reply_parents:
            value = allocate('comment', old, comment_id(reply_parents[old]) + '.', 22)
        else:
            value = allocate('comment', old, 'Ugx', 23)
        active.remove(old)
        mappings['comment'][old] = value
        return value

    for old in sorted(namespaces['comment']):
        comment_id(old)
    for old in sorted(namespaces['video']):
        mappings['video'][old] = allocate('video', old)
    for old in sorted(namespaces['channel']):
        mappings['channel'][old] = allocate('channel', old, 'UC', 22)
    for row in synthetic:
        for key in ID_COLUMNS:
            if row[key]:
                namespace = 'video' if key == 'video_id' else 'channel' if key == 'author_channel_id' else 'comment'
                row[key] = mappings[namespace][row[key]]
    for row in video_rows:
        row['video_id'] = mappings['video'][row['video_id']]
        if row['channel_id']:
            row['channel_id'] = mappings['channel'][row['channel_id']]
    result = pa.Table.from_pylist(rows, schema=table.schema)
    if len(set(result['comment_id'].to_pylist())) != len(result):
        raise ValueError('Duplicate comment ID after rewrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.remap-ids-', dir=output.parent))
    try:
        pq.write_table(result, temp / 'comments.parquet', compression='zstd')
        (temp / 'videos').mkdir()
        pq.write_table(pa.Table.from_pylist(video_rows, schema=video_table.schema), temp / 'videos/part-00000.parquet', compression='zstd')
        with (temp / 'comments.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=table.column_names, lineterminator='\n')
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
        with (temp / 'id_mapping.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['namespace', 'old_id', 'new_id'])
            for namespace, mapping in mappings.items():
                writer.writerows((namespace, old, new) for old, new in sorted(mapping.items()))
        after = {"comments": file_hash(comments), "videos": metadata_hash(videos)}
        if before != after:
            raise RuntimeError('Input changed during rewrite')
        summary = {
            'rows': len(result), 'columns': result.column_names,
            'source_counts': dict(Counter(result['source'].to_pylist())),
            'synthetic_rows_with_remapped_ids': len(synthetic),
            'changed_columns': list(ID_COLUMNS), 'seed_method': 'SHA256/ytb-simulation-id-v1',
            'mapping_counts': {k: len(v) for k, v in mappings.items()},
            'input_hashes_before': before, 'input_hashes_after': after,
            'source_preserved': True, 'text_and_dates_preserved': True,
            'identifiers_are_fictional': True,
            'note': 'Format-compatible simulation IDs, not verified YouTube entities. Original source values retained. Mapping report intentionally contains original SYN_ IDs.',
        }
        (temp / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comments', type=Path, default=Path('data/mixed_comments_73247_v2/comments.parquet'))
    parser.add_argument('--videos', type=Path, default=Path('data/synthetic_comments_50000/videos'))
    parser.add_argument('--real-core', type=Path, default=Path('data/core'))
    parser.add_argument('--output', type=Path, default=Path('data/mixed_comments_73247_v3'))
    args = parser.parse_args()
    print(json.dumps(rewrite(args.comments, args.videos, args.real_core, args.output), ensure_ascii=False, indent=2))
