"""Train GSDMM from the user's filtered, seven-column Silver Parquet table."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

K, ALPHA, BETA = 20, 0.1, 0.1
MIN_WORD_FREQ = 2
CAP_PER_VIDEO = None


def token_list(value):
    # pandas/pyarrow commonly returns list columns as numpy arrays.
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        value = value.split()
    if not isinstance(value, (list, tuple)):
        return []
    return [word for word in value if isinstance(word, str) and word.strip()]


def clean_for_dedup(value):
    # Recreate the text key dropped by the user's Silver processing code.
    value = str(value).lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"(.)\1{2,}", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--input", type=Path, default=Path("/opt/data/silver/comments_clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/data/gold"))
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.seed < 0 or args.iterations < 1:
        parser.error("seed >= 0; iterations >= 1")

    source, destination = args.input.resolve(), args.output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Output phải tách biệt với dữ liệu Silver đầu vào")
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    output_names = [f"comments_clustered{suffix}.parquet", f"gsdmm_model{suffix}.npz",
                    f"cluster_top_words{suffix}.csv", f"training_summary{suffix}.json"]
    conflicts = [name for name in output_names if (args.output_dir / name).exists()]
    if conflicts:
        parser.error(f"Đã có kết quả {conflicts}; dùng --output-dir khác để giữ lần chạy cũ")

    files = sorted(args.input.glob("*.parquet")) if args.input.is_dir() else [args.input]
    if not files or any(not path.is_file() for path in files):
        parser.error(f"Không tìm thấy Parquet: {args.input}")
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    print(f"Đã đọc {len(files)} file Parquet, {len(raw):,} dòng.", flush=True)
    print("Các cột:", list(raw.columns), flush=True)
    missing = {"comment_id", "video_id", "text", "tokens"} - set(raw.columns)
    if missing:
        parser.error(f"Thiếu cột: {sorted(missing)}")
    funnel = [("raw", len(raw))]
    df = raw.loc[
        raw["comment_id"].notna() & raw["video_id"].notna() & raw["text"].notna()
    ].copy()
    df = df.loc[df["text"].astype(str).str.strip().ne("")]
    df = df.sort_values("comment_id", kind="stable").drop_duplicates("comment_id").copy()
    funnel.append(("bỏ rỗng/trùng ID", len(df)))

    # Silver is already filtered; do not apply an additional >= 3 token gate.
    df["_tokens"] = df["tokens"].map(token_list)

    # Optional compatibility if a future Silver table retains this flag.
    if "is_reply" in df.columns:
        df = df.loc[~df["is_reply"].fillna(False).astype(bool)].copy()
        funnel.append(("bỏ reply", len(df)))
    else:
        print("Không có is_reply: dùng Silver đã bỏ reply ở bước trước.", flush=True)

    if "text_clean" not in df.columns:
        df["text_clean"] = df["text"].map(clean_for_dedup)
    df = df.loc[df["text_clean"].notna() & df["text_clean"].str.strip().ne("")]
    df = df.drop_duplicates("text_clean").copy()
    funnel.append(("bỏ trùng text_clean", len(df)))
    print("Không lọc lại theo dấu tiếng Việt; dùng kết quả lọc Silver.", flush=True)

    if CAP_PER_VIDEO:
        df = pd.concat([
            group.sample(min(len(group), CAP_PER_VIDEO), random_state=0)
            for _, group in df.groupby("video_id", sort=True)
        ])
        funnel.append((f"cap {CAP_PER_VIDEO}/video", len(df)))
    df = df.sort_values("comment_id", kind="stable").reset_index(drop=True)
    if df.empty:
        parser.error(f"Không còn bình luận đủ điều kiện: {funnel}")

    docs = df["_tokens"].tolist()
    freq = Counter(word for doc in docs for word in doc)
    words = sorted(word for word, count in freq.items() if count >= MIN_WORD_FREQ)
    vocab = {word: index for index, word in enumerate(words)}
    docs_ids, keep = [], []
    for index, doc in enumerate(docs):
        counts = Counter(word for word in doc if word in vocab)
        if sum(counts.values()) >= 2:
            docs_ids.append((np.array([vocab[word] for word in counts], dtype=np.int64),
                             np.array(list(counts.values()), dtype=np.int64)))
            keep.append(index)
    df = df.iloc[keep].copy().reset_index(drop=True)
    D, V = len(docs_ids), len(vocab)
    funnel.append(("sau lọc từ hiếm", D))
    if D == 0 or V == 0:
        parser.error(f"Không đủ bình luận/từ để train: {funnel}")
    df["text_gsdmm"] = df["_tokens"].map(" ".join)
    vc = df["video_id"].value_counts()
    for label, count in funnel:
        print(f"{label:26s} {count}", flush=True)
    print("Top 1 video chiếm:", round(vc.iloc[0] / D * 100, 1), "%")
    print("Top 10 video chiếm:", round(vc.head(10).sum() / D * 100, 1), "%")
    print(f"Số video: {len(vc)}, số bình luận train: {D}, từ điển: {V}", flush=True)
    print("Mẫu tokens:")
    for text in df["text_gsdmm"].sample(min(5, D), random_state=1):
        print("  ", text[:100])

    # GSDMM parameters and Gibbs update equations are unchanged.
    rng = np.random.default_rng(args.seed)
    z = rng.integers(0, K, size=D)
    m = np.zeros(K, dtype=np.int64)
    n = np.zeros(K, dtype=np.int64)
    nkw = np.zeros((K, V), dtype=np.int64)
    for d, (ids, counts) in enumerate(docs_ids):
        k = z[d]
        m[k] += 1
        n[k] += counts.sum()
        nkw[k, ids] += counts
    history = []
    for iteration in range(args.iterations):
        moved = 0
        for d in rng.permutation(D):
            ids, counts = docs_ids[d]
            old = z[d]
            m[old] -= 1
            n[old] -= counts.sum()
            nkw[old, ids] -= counts
            lp = np.log(m + ALPHA)
            for word, count in zip(ids, counts):
                for j in range(int(count)):
                    lp += np.log(nkw[:, word] + BETA + j)
            for j in range(int(counts.sum())):
                lp -= np.log(n + V * BETA + j)
            probabilities = np.exp(lp - lp.max())
            probabilities /= probabilities.sum()
            new = min(int(np.searchsorted(np.cumsum(probabilities), rng.random())), K - 1)
            z[d] = new
            m[new] += 1
            n[new] += counts.sum()
            nkw[new, ids] += counts
            moved += int(new != old)
        history.append({"iteration": iteration + 1, "moved": moved,
                        "active_topics": int((m > 0).sum())})
        print(f"Vòng {iteration + 1}/{args.iterations}: {moved} bình luận đổi cụm, "
              f"còn {(m > 0).sum()} cụm", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["_tokens"])
    df["cluster"] = z
    df.to_parquet(args.output_dir / f"comments_clustered{suffix}.parquet", index=False)
    np.savez_compressed(args.output_dir / f"gsdmm_model{suffix}.npz", m=m, n=n, nkw=nkw,
                        vocab=np.array(words), seed=args.seed, k=K, alpha=ALPHA,
                        beta=BETA, iterations=args.iterations, min_word_freq=MIN_WORD_FREQ)
    rows = []
    col_tot = nkw.sum(axis=0) + 1
    for k in np.argsort(-m, kind="stable"):
        if m[k] == 0:
            continue
        score = (nkw[k] / col_tot) * np.log1p(nkw[k])
        top = [words[i] for i in np.argsort(-score, kind="stable") if nkw[k, i] > 0][:15]
        rows.append({"cluster": int(k), "so_binh_luan": int(m[k]), "top_words": " ".join(top)})
        print(f"\n=== Cụm {k} ({int(m[k])} bình luận) ===")
        print("Từ khóa:", " ".join(top))
        for text in df.loc[df["cluster"] == k, "text"].head(3):
            print("  -", str(text).replace("\n", " ")[:100])
    pd.DataFrame(rows).to_csv(args.output_dir / f"cluster_top_words{suffix}.csv",
                              index=False, encoding="utf-8-sig")
    report = {"input": str(args.input), "input_files": [path.name for path in files],
              "input_rows": len(raw), "input_columns": list(raw.columns),
              "funnel": dict(funnel), "training_rows": D, "vocabulary_size": V,
              "seed": args.seed, "k": K, "alpha": ALPHA, "beta": BETA,
              "iterations": args.iterations, "min_tokens": None,
              "min_tokens_after_vocabulary_filter": 2,
              "min_word_freq": MIN_WORD_FREQ, "cap_per_video": CAP_PER_VIDEO,
              "accent_language_filter": False, "history": history}
    (args.output_dir / f"training_summary{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu kết quả tại: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
