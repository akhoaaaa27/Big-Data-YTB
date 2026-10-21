"""Merge already filtered comments without changing their 18 Core columns.

Run from the repository root after extracting source_comparison into data/:
    python datasets/combined_filtered_comments/combine_filtered.py
Then train the combined topics input using --no-vi-filter.
"""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def fingerprint(path):
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*.parquet")):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def combine(source, output, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output must use separate directories")
    if output.exists():
        raise FileExistsError(output)
    before = fingerprint(source)
    # Load and check both tasks before creating any output.
    tables, counts = {}, {}
    schema = None
    for task in ("topics", "sentiment"):
        parts = [pq.read_table(source / origin / task / "comments/part-00000.parquet")
                 for origin in ("real", "synthetic")]
        schema = schema or parts[0].schema
        if len(schema) != 18 or not all(schema.equals(t.schema, check_metadata=True) for t in parts):
            raise ValueError("Both inputs must have the same 18-column Core schema")
        merged = pa.concat_tables(parts)
        ids = merged["comment_id"].to_pylist()
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("comment_id must be nonempty and unique")
        indices = list(range(len(merged)))
        random.Random(seed).shuffle(indices)
        tables[task] = merged.take(pa.array(indices, type=pa.int64()))
        counts[task] = {
            "rows": len(merged), "columns": len(schema),
            "source_counts": dict(Counter(merged["source"].to_pylist())),
            "duplicate_text_clean": len(merged) - len(set(merged["text_clean"].to_pylist())),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".combined-", dir=output.parent))
    try:
        for task, table in tables.items():
            target = temporary / task / "comments"
            target.mkdir(parents=True)
            pq.write_table(table, target / "part-00000.parquet", compression="zstd")
        after = fingerprint(source)
        if before != after:
            raise RuntimeError("Input data changed during merge")
        summary = {
            "input": str(source), "output": str(output), "seed": seed,
            "input_sha256_before": before, "input_sha256_after": after,
            "tasks": counts, "columns": schema.names,
            "values_preserved": True, "date_values_preserved": True,
            "new_origin_columns": False,
            "training": "One combined model; global text deduplication and rare-word filtering happen in train_topics",
            "limitation": "Mixed real and synthetic data; topic prevalence is not evidence about real audiences alone.",
        }
        (temporary / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/source_comparison"))
    parser.add_argument("--output", type=Path, default=Path("data/combined_filtered_comments"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(combine(args.input, args.output, args.seed), ensure_ascii=False, indent=2))
