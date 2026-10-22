"""Name clusters from the reviewed 34,923-comment run and label each comment.

These are cluster-level, provisional aspect labels, not per-comment predictions.
Unrelated or unclear clusters map to Khác. Sentiment is not assigned here.
"""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_MAPPING = Path(__file__).resolve().parent.parent / "config/cluster_aspects_seed42.json"


def read_words(words_path):
    with Path(words_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [{"cluster": int(row["cluster"]), "so_binh_luan": int(row["so_binh_luan"]),
                 "top_words": " ".join(row["top_words"].split())}
                for row in csv.DictReader(stream)]
    if not rows or len({row["cluster"] for row in rows}) != len(rows):
        raise ValueError("Bảng từ khóa phải có mỗi cụm đúng một dòng")
    if any(row["cluster"] < 0 or row["so_binh_luan"] < 1 for row in rows):
        raise ValueError("cluster >= 0 và so_binh_luan >= 1")
    rows.sort(key=lambda row: row["cluster"])
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def prepare_mapping(words_path, destination):
    """Tạo bảng để đặt tên CHO CỤM; không gán nhãn tay từng bình luận."""
    rows, signature = read_words(words_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {"version": 1, "method": "provisional_cluster_level_mapping",
             "cluster_words_signature": signature,
             "note": "Mặc định Khác. Xem từ khóa và bình luận mẫu của mỗi cụm rồi điền một tên vào cluster_aspects.",
             "cluster_aspects": {str(row["cluster"]): "Khác" for row in rows},
             "review_context": rows}
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return destination


def load_mapping(mapping_path, signature, clusters):
    with Path(mapping_path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("cluster_words_signature") != signature:
        raise ValueError("Bảng từ khóa khác lần train đã lập nhãn. Dùng --prepare-mapping để tạo bảng mới, xem lại tên cụm rồi chạy với --mapping.")
    labels = value.get("cluster_aspects")
    if not isinstance(labels, dict) or set(labels) != {str(cluster) for cluster in clusters}:
        raise ValueError("cluster_aspects phải có đúng các mã cụm của lần train này")
    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise ValueError("Mỗi cụm phải có đúng một tên khía cạnh dạng chuỗi không rỗng")
    return {int(cluster): label.strip() for cluster, label in labels.items()}


def label_comments(input_path, words_path, output, mapping_path=DEFAULT_MAPPING):
    input_path, words_path, output = map(Path, (input_path, words_path, output))
    output = output.resolve()
    for path in (input_path, words_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {path}")
        source = path.resolve()
        if source == output or output in source.parents or source in output.parents:
            raise ValueError("Thư mục output phải tách biệt với file đầu vào")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Đã có {output}; chọn --output-dir khác")
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    word_rows, signature = read_words(words_path)
    cluster_aspects = load_mapping(mapping_path, signature, [row["cluster"] for row in word_rows])
    table = pq.read_table(input_path)
    missing = {"comment_id", "cluster"} - set(table.column_names)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")
    if "aspect" in table.column_names:
        raise ValueError("Bảng đã có aspect; dùng comments_clustered.parquet gốc")
    ids = table["comment_id"].to_pylist()
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("comment_id phải duy nhất và không rỗng")
    if not pa.types.is_integer(table.schema.field("cluster").type):
        raise ValueError("cluster phải có kiểu số nguyên")
    clusters = table["cluster"].to_pylist()
    expected_counts = {row["cluster"]: row["so_binh_luan"] for row in word_rows}
    if dict(Counter(clusters)) != expected_counts:
        raise ValueError("Số bình luận từng cụm không khớp cluster_top_words.csv; hãy dùng hai file cùng lần train")
    if set(clusters) - set(cluster_aspects):
        raise ValueError("Có cụm chưa nằm trong bảng nhãn")
    aspects = [cluster_aspects[cluster] for cluster in clusters]
    result = table.append_column("aspect", pa.array(aspects, type=pa.string()))
    counts = Counter(aspects)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".assign-aspects-", dir=output.parent))
    try:
        pq.write_table(result, temp / "comments_with_aspects.parquet", compression="zstd")
        with (temp / "comments_with_aspects.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result.column_names, lineterminator="\n")
            writer.writeheader()
            for row in result.to_pylist():
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                                 for key, value in row.items()})
        with (temp / "cluster_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["cluster", "aspect", "so_binh_luan", "top_words"], lineterminator="\n")
            writer.writeheader()
            writer.writerows({**row, "aspect": cluster_aspects[row["cluster"]]} for row in word_rows)
        with (temp / "aspect_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["aspect", "so_binh_luan", "ty_le_phan_tram"], lineterminator="\n")
            writer.writeheader()
            writer.writerows({"aspect": aspect, "so_binh_luan": count,
                              "ty_le_phan_tram": round(count / len(result) * 100, 2)}
                             for aspect, count in counts.most_common())
        after = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if after != before:
            raise RuntimeError("File đầu vào thay đổi trong lúc gán nhãn")
        summary = {"input": str(input_path.resolve()), "rows": len(result),
                   "aspect_counts": dict(counts), "other_rows": counts.get("Khác", 0),
                   "input_sha256_before": before, "input_sha256_after": after,
                   "cluster_words_signature": signature,
                   "mapping": str(Path(mapping_path).resolve()),
                   "method": "provisional_cluster_level_mapping",
                   "sentiment_assigned": False,
                   "note": "All members inherit their cluster's label; mixed clusters can contain individual mismatches. Labels are not independently validated per comment."}
        (temp / "label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/opt/data/gold/comments_clustered.parquet"))
    parser.add_argument("--top-words", type=Path, default=Path("/opt/data/gold/cluster_top_words.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/data/gold/aspects_seed42"))
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                        help="JSON: một tên khía cạnh cho mỗi cụm GSDMM")
    parser.add_argument("--prepare-mapping", type=Path,
                        help="Chỉ tạo JSON tên cụm mặc định Khác cho bảng từ khóa hiện tại; chưa gán nhãn")
    args = parser.parse_args()
    try:
        if args.prepare_mapping:
            path = prepare_mapping(args.top_words, args.prepare_mapping)
            print(f"Đã tạo {path}. Điền tên trong cluster_aspects sau khi xem từng cụm.", flush=True)
        else:
            summary = label_comments(args.input, args.top_words, args.output_dir, args.mapping)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    if not args.prepare_mapping:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Đã lưu bảng gán khía cạnh tại: {args.output_dir}", flush=True)
