"""Dự đoán sentiment bằng mô hình Transformers do người chạy chỉ định.

python -m ytb.modeling.predict_sentiment --model TEN_MODEL_HOAC_DUONG_DAN

Cần cài torch và transformers trong môi trường chạy (không cài/tải lúc import).
Dùng model nhận văn bản tiếng Việt nguyên câu; module không tự tách từ cho
những model cần word segmentation. Mặc định đọc text_clean, giữ emoji/phủ định,
không dùng tokens đã bỏ từ dừng và không loại câu ngắn/reply vì is_modelable.

Nhãn được đọc từ id2label. Nếu model dùng nhãn mơ hồ, cần chỉ rõ, ví dụ:
--label-map "LABEL_0=negative,LABEL_1=positive,LABEL_2=neutral"
Thứ tự ví dụ KHÔNG phải quy ước chung cho mọi model; phải kiểm tra model card.

--topics PATH ghép kết quả GSDMM theo comment_id, giữ mọi bình luận sentiment.
Sentiment là cảm xúc chung của câu, không tự suy ra thái độ ủng hộ/phản đối AI.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .. import config

LABELS = ("negative", "neutral", "positive")
ALIASES = {"neg": "negative", "negative": "negative", "neu": "neutral", "neutral": "neutral",
           "pos": "positive", "positive": "positive"}


def _mapping(classifier, explicit=None) -> dict[str, str]:
    mapping = dict(explicit or {})
    if not mapping:
        labels = classifier.model.config.id2label.values()
        mapping = {str(label): ALIASES.get(str(label).casefold()) for label in labels}
    if len(mapping) != 3 or set(mapping.values()) != set(LABELS):
        raise ValueError("Cần ánh xạ chính xác 3 nhãn negative/neutral/positive. Dùng --label-map theo model card, không đoán thứ tự nhãn.")
    return mapping


def predict(source: Path, output: Path, *, model_name: str, text_column="text_clean", topics=None,
            batch_size=16, max_length=256, device=-1, label_map=None, classifier=None) -> dict:
    source, output = Path(source), Path(output)
    if not model_name:
        raise ValueError("Cần chỉ định mô hình sentiment")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size và max_length phải > 0")
    paths = [source.resolve()] + ([Path(topics).resolve()] if topics else [])
    for path in paths:
        if path == output.resolve() or path in output.resolve().parents or output.resolve() in path.parents:
            raise ValueError("Thư mục đầu ra phải tách biệt đường dẫn dữ liệu đầu vào")
    if output.exists():
        raise FileExistsError(f"{output} đã tồn tại. Chọn thư mục khác để giữ kết quả cũ.")
    table = ds.dataset(source, format="parquet").to_table()
    missing = {"comment_id", "video_id", text_column} - set(table.column_names)
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(sorted(missing))}")
    ids = table["comment_id"].to_pylist()
    if any(not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("comment_id phải duy nhất và không rỗng")
    texts = table[text_column].to_pylist()
    if any(t is not None and not isinstance(t, str) for t in texts):
        raise ValueError(f"{text_column} phải là văn bản")
    indices = [i for i, text in enumerate(texts) if text and text.strip()]
    if not indices:
        raise ValueError("Không có bình luận chứa văn bản để dự đoán")
    added = {"sentiment", "sentiment_score", *(f"sentiment_{label}" for label in LABELS)}
    if added.intersection(table.column_names):
        raise ValueError("Đầu vào đã chứa cột sentiment; hãy dùng bình luận Core")
    clusters = None
    if topics:
        if "cluster" in table.column_names:
            raise ValueError("Đầu vào đã có cluster; bỏ --topics hoặc dùng bình luận Core")
        topics_table = ds.dataset(topics, format="parquet").to_table()
        if not {"comment_id", "video_id", "cluster"}.issubset(topics_table.column_names):
            raise ValueError("Bảng GSDMM phải có comment_id, video_id, cluster")
        topic_rows = topics_table.select(["comment_id", "video_id", "cluster"]).to_pylist()
        topic_ids = [r["comment_id"] for r in topic_rows]
        if any(not cid for cid in topic_ids) or len(set(topic_ids)) != len(topic_ids):
            raise ValueError("comment_id trong kết quả GSDMM phải duy nhất và không rỗng")
        source_videos = dict(zip(ids, table["video_id"].to_pylist()))
        if any(r["comment_id"] not in source_videos or source_videos[r["comment_id"]] != r["video_id"] for r in topic_rows):
            raise ValueError("Kết quả GSDMM không thuộc cùng tập bình luận đầu vào")
        lookup = {r["comment_id"]: r["cluster"] for r in topic_rows}
        clusters = pa.array([lookup.get(cid) for cid in ids], type=pa.int64())
    if classifier is None:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("Thiếu thư viện sentiment. Cần cài torch và transformers trong môi trường chạy.") from error
        classifier = pipeline("text-classification", model=model_name, tokenizer=model_name,
                              device=device, trust_remote_code=False)
    mapping = _mapping(classifier, label_map)
    labels, confidence = [None] * len(ids), [None] * len(ids)
    probabilities = {label: [None] * len(ids) for label in LABELS}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        predictions = classifier([texts[i] for i in batch], top_k=None, batch_size=batch_size,
                                 truncation=True, max_length=max_length)
        if len(predictions) != len(batch):
            raise ValueError("Số kết quả sentiment không khớp số bình luận")
        for index, scores in zip(batch, predictions):
            by_label = {}
            for item in scores:
                label = mapping.get(str(item["label"]))
                score = float(item["score"])
                if label is None or label in by_label or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Nhãn hoặc điểm sentiment không hợp lệ")
                by_label[label] = score
            if set(by_label) != set(LABELS) or not math.isclose(sum(by_label.values()), 1, abs_tol=0.001):
                raise ValueError("Mô hình phải trả đủ phân phối xác suất 3 nhãn sentiment")
            label = max(LABELS, key=by_label.get)
            labels[index], confidence[index] = label, by_label[label]
            for name in LABELS:
                probabilities[name][index] = by_label[name]
        print(f"Sentiment: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    result = table.replace_schema_metadata(None).append_column("sentiment", pa.array(labels, type=pa.string()))
    result = result.append_column("sentiment_score", pa.array(confidence, type=pa.float32()))
    for label in LABELS:
        result = result.append_column(f"sentiment_{label}", pa.array(probabilities[label], type=pa.float32()))
    summary = {"source": str(source.resolve()), "model": model_name, "text_column": text_column,
               "label_map": mapping, "rows": len(ids), "predicted": len(indices), "empty_text": len(ids) - len(indices),
               "labels": dict(Counter(label for label in labels if label is not None)),
               "max_length": max_length, "truncation": True, "topics": str(Path(topics).resolve()) if topics else None,
               "with_topic": sum(v is not None for v in clusters.to_pylist()) if clusters is not None else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".sentiment-", dir=output.parent))
    try:
        pq.write_table(result, temp / "comments_sentiment.parquet", compression="zstd")
        if clusters is not None:
            pq.write_table(result.append_column("cluster", clusters), temp / "comments_topics_sentiment.parquet", compression="zstd")
        (temp / "sentiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"{output} xuất hiện trong lúc chạy; không ghi đè")
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.DATA_DIR / "core" / "comments")
    parser.add_argument("--output-dir", type=Path, default=config.DATA_DIR / "business" / "sentiment")
    parser.add_argument("--model", required=True, help="Tên Hugging Face hoặc đường dẫn model local đã chọn")
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--topics", type=Path, help="comments_clustered.parquet của cùng bộ đầu vào")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=int, default=-1, help="-1: CPU; 0: GPU CUDA đầu tiên")
    parser.add_argument("--label-map", help="Ánh xạ theo model card, ví dụ NEG=negative,NEU=neutral,POS=positive")
    args = parser.parse_args()
    try:
        mapping = None
        if args.label_map:
            pairs = [part.split("=", 1) for part in args.label_map.split(",")]
            if any(len(pair) != 2 for pair in pairs):
                raise ValueError("--label-map có dạng NHAN_MODEL=negative,NHAN_MODEL=neutral,NHAN_MODEL=positive")
            mapping = {key.strip(): value.strip().lower() for key, value in pairs}
            if len(mapping) != len(pairs):
                raise ValueError("--label-map bị trùng nhãn model")
        summary = predict(args.input, args.output_dir, model_name=args.model, text_column=args.text_column,
                          topics=args.topics, batch_size=args.batch_size, max_length=args.max_length,
                          device=args.device, label_map=mapping)
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(json.dumps({**summary, "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
