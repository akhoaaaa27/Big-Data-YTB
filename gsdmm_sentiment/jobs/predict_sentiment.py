"""Thêm sentiment chung của bình luận, giữ nguyên cluster/aspect từ GSDMM.

Default model: 5CD-AI/Vietnamese-Sentiment-visobert (raw Vietnamese text).
Nhãn đọc từ config; không đoán LABEL_0/1/2. Không ép câu Khác thành trung lập.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MODEL = "5CD-AI/Vietnamese-Sentiment-visobert"
LABELS = ("negative", "neutral", "positive")
VI_LABELS = {"negative": "Tiêu cực", "neutral": "Trung lập", "positive": "Tích cực"}
ALIASES = {"negative": "negative", "neg": "negative", "tiêu cực": "negative",
           "neutral": "neutral", "neu": "neutral", "trung lập": "neutral",
           "positive": "positive", "pos": "positive", "tích cực": "positive"}


def model_labels(id2label, explicit=None):
    mapping = {}
    for index in range(3):
        label = str(id2label.get(index, id2label.get(str(index), "")))
        value = (explicit or {}).get(str(index), (explicit or {}).get(label))
        if value is None:
            value = ALIASES.get(label.casefold())
        if value not in LABELS:
            raise ValueError(f"Không rõ nhãn {index}={label!r}. Kiểm tra model card rồi dùng --label-map; không đoán thứ tự nhãn.")
        mapping[index] = value
    if set(mapping.values()) != set(LABELS):
        raise ValueError("Mô hình phải có đủ 3 nhãn negative/neutral/positive khác nhau")
    return mapping


def clean_for_sentiment(value):
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", text)
    # Không bỏ emoji, từ phủ định, stopword hoặc viết thường toàn bộ câu.
    return re.sub(r"\s+", " ", text).strip()


class SentimentModel:
    def __init__(self, model_name, revision, device, max_length, threads, label_map):
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as error:
            raise RuntimeError("Thiếu torch/transformers. Chạy bằng image sentiment-local đã build.") from error
        self.torch = torch
        self.device = device
        self.max_length = max_length
        if device == "cpu":
            torch.set_num_threads(threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA không có sẵn; dùng --device cpu")
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
        if config.num_labels != 3:
            raise ValueError("Cần mô hình phân loại 3 cảm xúc")
        self.mapping = model_labels(config.id2label, label_map)
        self.metadata = {"model": model_name, "revision": revision,
                         "resolved_commit": getattr(config, "_commit_hash", None),
                         "id2label": config.id2label, "label_mapping": self.mapping}
        print("Ánh xạ nhãn model:", self.mapping, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, config=config, trust_remote_code=False).to(device).eval()
        model_limit = getattr(self.tokenizer, "model_max_length", 10**9)
        if max_length > model_limit:
            raise ValueError(f"--max-length vượt giới hạn tokenizer: {model_limit}")

    def predict(self, texts):
        # Chỉ đếm độ dài ở đây; đầu vào model được cắt ở lần tokenize bên dưới.
        # verbose=False tránh cảnh báo dài quá giới hạn dù model không nhận chuỗi đó.
        lengths = self.tokenizer(texts, add_special_tokens=True, truncation=False,
                                 padding=False, verbose=False)["input_ids"]
        truncated = [len(ids) > self.max_length for ids in lengths]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits.float(), dim=-1).cpu().tolist()
        return [({self.mapping[i]: float(value) for i, value in enumerate(scores)}, flag)
                for scores, flag in zip(probabilities, truncated)]


def write_csv(path, table):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table.column_names, lineterminator="\n")
        writer.writeheader()
        for row in table.to_pylist():
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})


def predict(source, output, *, model_name=MODEL, revision="main", batch_size=16,
            max_length=128, device="cpu", threads=4, limit=None, label_map=None, predictor=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không thấy file đầu vào: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output phải tách biệt với file đầu vào")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Đã có {output}; chọn --output-dir khác để giữ kết quả cũ")
    if batch_size < 1 or max_length < 1 or threads < 1 or (limit is not None and limit < 1):
        raise ValueError("batch-size/max-length/cpu-threads/limit phải > 0")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    full = pq.read_table(source)
    table = full.slice(0, min(limit, len(full))) if limit is not None else full
    missing = {"comment_id", "video_id", "aspect", "text"} - set(table.column_names)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}; hãy dùng bảng một khía cạnh aspects_seed42")
    added = {"sentiment", "sentiment_score", "sentiment_truncated", *(f"sentiment_{name}" for name in LABELS)}
    if added & set(table.column_names):
        raise ValueError("File đã có sentiment; hãy dùng bảng khía cạnh gốc")
    ids = table["comment_id"].to_pylist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("comment_id phải duy nhất, không rỗng")
    texts = [clean_for_sentiment(text) for text in table["text"].to_pylist()]
    indices = [i for i, text in enumerate(texts) if text]
    if not indices:
        raise ValueError("Không có bình luận chứa văn bản để dự đoán")
    print(f"Đã đọc {len(full):,} bình luận; xử lý {len(table):,} dòng, {len(indices):,} câu có văn bản", flush=True)
    if predictor is None:
        print(f"Đang nạp mô hình {model_name}; lần đầu cần Internet để tải model", flush=True)
        predictor = SentimentModel(model_name, revision, device, max_length, threads, label_map)
    metadata = getattr(predictor, "metadata", {"model": model_name, "revision": revision})
    labels, confidence = [None] * len(table), [None] * len(table)
    truncated = [False] * len(table)
    scores = {name: [None] * len(table) for name in LABELS}
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        predictions = predictor.predict([texts[i] for i in selected])
        if len(predictions) != len(selected):
            raise ValueError("Số dự đoán không khớp batch")
        for index, (probabilities, cut) in zip(selected, predictions):
            if set(probabilities) != set(LABELS):
                raise ValueError("Mô hình phải trả đủ 3 nhãn")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001):
                raise ValueError("Điểm xác suất không hợp lệ")
            winner = max(LABELS, key=probabilities.get)
            labels[index], confidence[index] = VI_LABELS[winner], probabilities[winner]
            truncated[index] = bool(cut)
            for name in LABELS:
                scores[name][index] = probabilities[name]
        done = min(start + batch_size, len(indices))
        if done == len(indices) or done % 256 < batch_size:
            print(f"Sentiment: {done:,}/{len(indices):,}", flush=True)
    result = table.append_column("sentiment", pa.array(labels, type=pa.string()))
    result = result.append_column("sentiment_score", pa.array(confidence, type=pa.float32()))
    for name in LABELS:
        result = result.append_column(f"sentiment_{name}", pa.array(scores[name], type=pa.float32()))
    result = result.append_column("sentiment_truncated", pa.array(truncated))
    aspect_values = table["aspect"].to_pylist()
    group_counts = Counter(zip(aspect_values, labels))
    totals = Counter(aspect_values)
    stats = pa.Table.from_pylist([
        {"aspect": aspect, "sentiment": sentiment or "Chưa dự đoán", "comments": count,
         "percent_within_aspect": round(count / totals[aspect] * 100, 2)}
        for (aspect, sentiment), count in group_counts.items()])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".sentiment-", dir=output.parent))
    try:
        pq.write_table(result, temp / "comments_sentiment.parquet", compression="zstd")
        write_csv(temp / "comments_sentiment.csv", result)
        write_csv(temp / "aspect_sentiment_summary.csv", stats)
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError("File khía cạnh đầu vào thay đổi trong lúc chạy")
        report = {"input": str(source), "input_rows": len(full), "output_rows": len(table),
                  "predicted": len(indices), "empty_text": len(table) - len(indices),
                  "sentiment_counts": dict(Counter(label for label in labels if label)),
                  "model": metadata, "max_length": max_length, "truncated_comments": sum(truncated),
                  "device": device, "limit": limit, "input_sha256_before": before, "input_sha256_after": after,
                  "note": "Whole-comment sentiment, not stance toward AI or guaranteed aspect-specific sentiment. Scores are softmax model outputs, not calibrated accuracy. All source columns/aspects, including Khác, retained. No independent accuracy evaluation on this dataset."}
        (temp / "sentiment_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/opt/data/gold/aspects_seed42/comments_with_aspects.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/data/gold/sentiment_seed42"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, help="Chạy thử N dòng đầu; chọn output khác bản đầy đủ")
    parser.add_argument("--label-map", help="Ánh xạ ID/tên nhãn model sang negative/neutral/positive theo model card")
    args = parser.parse_args()
    try:
        explicit = None
        if args.label_map:
            pairs = [part.split("=", 1) for part in args.label_map.split(",")]
            if any(len(pair) != 2 for pair in pairs):
                raise ValueError("--label-map phải gồm các cặp NHÃN=negative/neutral/positive")
            explicit = {key.strip(): value.strip().lower() for key, value in pairs}
            if len(explicit) != len(pairs):
                raise ValueError("--label-map bị trùng khóa")
        report = predict(args.input, args.output_dir, model_name=args.model, revision=args.revision,
                         batch_size=args.batch_size, max_length=args.max_length, device=args.device,
                         threads=args.cpu_threads, limit=args.limit, label_map=explicit)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, OSError) as error:
        parser.exit(1, f"Lỗi sentiment: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Đã lưu sentiment tại: {args.output_dir}", flush=True)
