"""GSDMM dựa trên gsdmm_train.py, dùng numpy và pyarrow đã có trong project.

Chạy từ thư mục repository:
    python -m ytb.modeling.train_topics --seed 42

Docker có mount data vào /opt/data:
    python -m ytb.modeling.train_topics --input /opt/data/core/comments \
        --output-dir /opt/data/business/topics/seed_42 --seed 42

Mặc định giữ các bước chọn mẫu của file gốc: is_modelable, bỏ reply, khử trùng
text_clean, heuristic ký tự tiếng Việt, tần suất từ >= 2. Heuristic này KHÔNG xác
minh ngôn ngữ, video nhạc AI hay xuất xứ Việt Nam. Không sửa dữ liệu đầu vào.
Đầu ra chỉ chứa tập bình luận thực sự dùng để huấn luyện, không phải toàn bộ Core.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .. import config

VI_CHARS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
REQUIRED = {"comment_id", "video_id", "is_modelable", "is_reply", "text_raw", "text_clean", "tokens"}


def _validate_paths(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Thư mục đầu ra phải tách biệt đường dẫn dữ liệu đầu vào")
    if output.exists():
        raise FileExistsError(f"{output} đã tồn tại. Chọn thư mục khác để giữ kết quả cũ.")


def _tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if not isinstance(value, (list, tuple)) or any(not isinstance(w, str) for w in value):
        raise ValueError("tokens phải là danh sách chuỗi hoặc chuỗi cách nhau bằng khoảng trắng")
    return [w for w in value if w]


def train(source: Path, output: Path, *, seed=42, k=20, alpha=0.1, beta=0.1,
          iterations=30, min_word_freq=2, cap_per_video=None, vi_filter=True) -> dict:
    source, output = Path(source), Path(output)
    _validate_paths(source, output)
    if k < 1 or iterations < 1 or min_word_freq < 1 or seed < 0:
        raise ValueError("k, iterations, min_word_freq phải > 0 và seed phải >= 0")
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        raise ValueError("alpha và beta phải là số hữu hạn > 0")
    if cap_per_video is not None and cap_per_video < 1:
        raise ValueError("cap_per_video phải > 0")
    table = ds.dataset(source, format="parquet").to_table()
    missing = REQUIRED - set(table.column_names)
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(sorted(missing))}")
    rows = table.to_pylist()
    ids = [r["comment_id"] for r in rows]
    if any(not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("comment_id phải duy nhất và không rỗng; cần khử trùng Core trước")
    if any(not row["video_id"] for row in rows):
        raise ValueError("video_id không được rỗng")
    funnel = {"input": len(rows)}
    # Sắp theo ID để seed cho kết quả ổn định khi đổi cách chia file Parquet.
    indices = sorted(range(len(rows)), key=lambda i: str(rows[i]["comment_id"]))
    indices = [i for i in indices if rows[i]["is_modelable"] is True]
    funnel["modelable"] = len(indices)
    indices = [i for i in indices if rows[i]["is_reply"] is False]
    funnel["without_replies"] = len(indices)
    seen = set()
    unique = []
    for i in indices:
        text = rows[i]["text_clean"]
        if isinstance(text, str) and text.strip() and text not in seen:
            seen.add(text)
            unique.append(i)
    indices = unique
    funnel["unique_text"] = len(indices)
    language_method = "disabled"
    if vi_filter:
        if "comment_language_status" in table.column_names:
            indices = [i for i in indices if rows[i]["comment_language_status"] == "vi"]
            language_method = "prepared_language_status"
        else:
            indices = [i for i in indices if any(c in VI_CHARS for c in str(rows[i]["text_raw"] or "").lower())]
            language_method = "legacy_accent_heuristic"
    funnel["after_vi_heuristic"] = len(indices)
    if cap_per_video:
        groups = defaultdict(list)
        for i in indices:
            groups[rows[i]["video_id"]].append(i)
        sampler = np.random.default_rng(0)  # Cùng tập huấn luyện cho các seed mô hình.
        indices = sorted((int(i) for group in groups.values()
                          for i in sampler.choice(group, min(len(group), cap_per_video), replace=False)),
                         key=lambda i: str(rows[i]["comment_id"]))
    funnel["after_video_cap"] = len(indices)
    docs = [_tokens(rows[i]["tokens"]) for i in indices]
    frequency = Counter(w for document in docs for w in document)
    words = sorted(w for w, count in frequency.items() if count >= min_word_freq)
    vocab = {w: i for i, w in enumerate(words)}
    documents, kept_indices, kept_tokens = [], [], []
    for index, document in zip(indices, docs):
        counts = Counter(w for w in document if w in vocab)
        if sum(counts.values()) >= 2:
            documents.append((np.array([vocab[w] for w in counts], dtype=np.int64),
                              np.array(list(counts.values()), dtype=np.int64)))
            kept_indices.append(index)
            kept_tokens.append(document)
    d_count, v_count = len(documents), len(vocab)
    funnel["training"] = d_count
    if not d_count or not v_count:
        raise ValueError(f"Không còn bình luận đủ điều kiện huấn luyện: {funnel}")
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, k, size=d_count)
    m, n, nkw = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64), np.zeros((k, v_count), dtype=np.int64)
    for topic, (word_ids, counts) in zip(assignments, documents):
        m[topic] += 1
        n[topic] += counts.sum()
        nkw[topic, word_ids] += counts
    history = []
    for iteration in range(iterations):
        moved = 0
        for index in rng.permutation(d_count):
            word_ids, counts = documents[index]
            old = assignments[index]
            m[old] -= 1
            n[old] -= counts.sum()
            nkw[old, word_ids] -= counts
            log_p = np.log(m + alpha)
            for word, count in zip(word_ids, counts):
                for j in range(int(count)):
                    log_p += np.log(nkw[:, word] + beta + j)
            for j in range(int(counts.sum())):
                log_p -= np.log(n + v_count * beta + j)
            probabilities = np.exp(log_p - log_p.max())
            probabilities /= probabilities.sum()
            new = min(int(np.searchsorted(np.cumsum(probabilities), rng.random())), k - 1)
            assignments[index] = new
            m[new] += 1
            n[new] += counts.sum()
            nkw[new, word_ids] += counts
            moved += int(new != old)
        history.append({"iteration": iteration + 1, "moved": moved, "active_topics": int((m > 0).sum())})
        print(f"Vòng {iteration + 1}/{iterations}: {moved} đổi cụm, {(m > 0).sum()} cụm có dữ liệu", flush=True)
    counts_by_video = Counter(rows[i]["video_id"] for i in kept_indices)
    largest = sorted(counts_by_video.values(), reverse=True)
    summary = {"source": str(source.resolve()), "seed": seed, "k": k, "alpha": alpha, "beta": beta,
               "iterations": iterations, "min_word_freq": min_word_freq,
               "vi_heuristic": language_method == "legacy_accent_heuristic",
               "language_filter_method": language_method,
               "cap_per_video": cap_per_video, "funnel": funnel, "vocabulary_size": v_count,
               "active_topics": int((m > 0).sum()), "videos": len(counts_by_video),
               "top_video_fraction": largest[0] / d_count, "top_10_video_fraction": sum(largest[:10]) / d_count,
               "history": history}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".topics-", dir=output.parent))
    try:
        selected = table.take(pa.array(kept_indices, type=pa.int64())).replace_schema_metadata(None)
        for name in ("cluster", "text_gsdmm"):
            if name in selected.column_names:
                raise ValueError(f"Đầu vào đã có cột {name}; hãy dùng bình luận Core")
        selected = selected.append_column("text_gsdmm", pa.array([" ".join(t) for t in kept_tokens]))
        selected = selected.append_column("cluster", pa.array(assignments, type=pa.int64()))
        pq.write_table(selected, temp / "comments_clustered.parquet", compression="zstd")
        np.savez_compressed(temp / "gsdmm_model.npz", m=m, n=n, nkw=nkw, vocab=np.array(words),
                            seed=seed, k=k, alpha=alpha, beta=beta, iterations=iterations)
        with (temp / "cluster_top_words.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cluster", "so_binh_luan", "top_words"], lineterminator="\n")
            writer.writeheader()
            totals = nkw.sum(axis=0) + 1
            for topic in np.argsort(-m, kind="stable"):
                if not m[topic]:
                    continue
                scores = nkw[topic] / totals * np.log1p(nkw[topic])
                top = [words[w] for w in np.argsort(-scores, kind="stable") if nkw[topic, w] > 0][:15]
                writer.writerow({"cluster": int(topic), "so_binh_luan": int(m[topic]), "top_words": " ".join(top)})
        (temp / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--min-word-freq", type=int, default=2)
    parser.add_argument("--cap-per-video", type=int)
    parser.add_argument("--no-vi-filter", action="store_true", help="Tắt lọc ngôn ngữ (cột prepare hoặc heuristic của đầu vào cũ)")
    args = parser.parse_args()
    target = args.output_dir or config.DATA_DIR / "business" / "topics" / f"seed_{args.seed}"
    try:
        result = train(args.input, target, seed=args.seed, k=args.k, alpha=args.alpha, beta=args.beta,
                       iterations=args.iterations, min_word_freq=args.min_word_freq,
                       cap_per_video=args.cap_per_video, vi_filter=not args.no_vi_filter)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(json.dumps({**result, "output": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
