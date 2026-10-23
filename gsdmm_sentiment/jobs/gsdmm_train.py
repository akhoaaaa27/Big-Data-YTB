"""Train GSDMM from the user's filtered, seven-column Silver Parquet table."""

import argparse
import json
import re
import hashlib
import tempfile 
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

FIXED_FILES = ( 
    "gsdmm_model.npz", "cluster_top_words.csv", 
    "cluster_aspects.json", "reviewed/comments_with_aspects.parquet", ) 

def file_sha256(path): 
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() 

def load_fixed_package(folder): 
    if __package__: 
        from .assign_aspects import read_words, load_mapping 
    else: 
        from assign_aspects import read_words, load_mapping 
        
    folder = Path(folder) 
    with np.load(folder / FIXED_FILES[0], allow_pickle=False) as saved: 
        m, n, nkw = (saved[name].copy() for name in ("m", "n", "nkw")) 
        words = saved["vocab"].tolist() 
        alpha, beta = float(saved["alpha"]), float(saved["beta"]) 
    
    if ( 
        m.ndim != 1 or n.shape != m.shape 
        or nkw.shape != (len(m), len(words)) 
        or len(set(words)) != len(words) or not words 
        or not all(np.isfinite(a).all() and (a >= 0).all() for a in (m, n, nkw)) 
        or not np.array_equal(nkw.sum(axis=1), n) 
        or not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0 ): 
        
        raise ValueError("Thông số model không hợp lệ.") 
    
    active = np.flatnonzero(m > 0) 
    rows, signature = read_words(folder / FIXED_FILES[1]) 
    labels = load_mapping(folder / FIXED_FILES[2], signature, active.tolist()) 
    if {r["cluster"]: r["so_binh_luan"] for r in rows} != { int(k): 
        int(m[k]) for k in active 
    }: 
        raise ValueError("Model và bảng từ khóa không cùng lần train.") 
    
    for row in rows: 
        k = row["cluster"] 
        score = (nkw[k] / (nkw.sum(axis=0) + 1)) * np.log1p(nkw[k]) 
        top = [words[i] for i in np.argsort(-score, kind="stable") if nkw[k, i] > 0][:15] 
        if " ".join(top) != row["top_words"]: 
            raise ValueError("Từ khóa không khớp model.") 
        
    reference = pd.read_parquet(folder / FIXED_FILES[3]) 
    required = {"comment_id", "video_id", "text", "tokens", "cluster", "aspect"} 
    if not required <= set(reference.columns): 
        raise ValueError("Thiếu cột trong kết quả train ban đầu.") 
    
    if ( 
        reference["comment_id"].isna().any() 
        or reference["comment_id"].astype(str).str.strip().eq("").any() 
        or reference["comment_id"].duplicated().any() 
        or dict(Counter(reference["cluster"].tolist())) 
        != {int(k): int(m[k]) for k in active} ): 
        
        raise ValueError("ID hoặc số dòng tham chiếu không khớp model.") 
    
    if any(row.aspect != labels.get(row.cluster) for row in reference.itertuples()): 
        raise ValueError("Nhãn tham chiếu không khớp bảng tên cụm.") 
    return m, n, nkw, words, alpha, beta, active, labels, reference 

def reference_key(row): 
    return json.dumps( 
        [str(row.video_id), str(row.text), token_list(row.tokens)], 
        ensure_ascii=False, ) 

def run_fixed_model(args, parser): 
    folder = args.model_dir.resolve() 
    manifest_path = folder / "model_manifest.json" 
    if args.mode != "seal": 
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) 
        
        if set(
            manifest.get("sha256", {})) != set(FIXED_FILES): 
            raise ValueError("Manifest thiếu file model.")

        for name in FIXED_FILES: 
            if file_sha256(folder / name) != manifest["sha256"][name]: 
                raise ValueError(f"File model đã thay đổi: {name}") 
    m, n, nkw, words, alpha, beta, active, labels, reference = load_fixed_package(folder) 
    
    if args.mode == "seal": 
        payload = {
            "version": folder.name, "sha256": {name: file_sha256(folder / name) for name in FIXED_FILES}} 
        
        with manifest_path.open("x", encoding="utf-8") as stream: 
            json.dump(payload, stream, ensure_ascii=False, indent=2) 
        
        print("Đã khóa bộ model:", folder, flush=True) 
        return 
    
    if args.mode == "check": 
        print("Bộ model hợp lệ:", folder, flush=True) 
        return 
    
    if args.output_dir is None: 
        parser.error("Cần --output-dir khi dự đoán.") 
    
    output = args.output_dir.resolve() 
    for protected in (folder, args.input.resolve()): 
        if output == protected or output in protected.parents or protected in output.parents: 
            raise ValueError("Output phải tách biệt model và dữ liệu đầu vào.") 
        
    if output.exists(): 
        raise FileExistsError(f"Đã có {output}; chọn thư mục mới.") 
        
    files = sorted(args.input.glob("*.parquet")) if args.input.is_dir() else [args.input] 
    if not files or any(not p.is_file() for p in files): 
        raise FileNotFoundError(f"Không có Parquet: {args.input}") 
        
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) 
    if not {"comment_id", "video_id", "text", "tokens"} <= set(df.columns): 
        raise ValueError("Cần Silver có comment_id, video_id, text, tokens.") 
        
    if {"cluster", "aspect", "sentiment"} & set(df.columns): 
        raise ValueError("Đầu vào phải là Silver chưa gán nhãn.") 
        
    input_rows = len(df) 
        
    df = df.dropna(subset=["comment_id", "video_id", "text"]).copy() 
    for col in ("comment_id", "video_id"): 
        df[col] = df[col].astype(str).str.strip() 

    df = df[df["comment_id"].ne("") & df["video_id"].ne("") & df["text"].astype(str).str.strip().ne("")] 
    if "is_reply" in df.columns: 
        df = df[~df["is_reply"].fillna(False).astype(bool)] 
            
    df = df.sort_values("comment_id", kind="stable").drop_duplicates("comment_id").copy() 
        
    if df.empty: 
        raise ValueError("Silver không có bình luận hợp lệ.") 
        
    df["tokens"] = df["tokens"].map(token_list) 
    if "text_clean" not in df.columns: 
        df["text_clean"] = df["text"].map(clean_for_dedup) 
        
    df["text_gsdmm"] = df["tokens"].map(" ".join) 
    baseline = {str(row.comment_id): (reference_key(row), int(row.cluster)) for row in reference.itertuples()} 
    vocab = {word: i for i, word in enumerate(words)} 
    assigned, counts = [], Counter() 
        
    for row in df.itertuples(): 
        old = baseline.get(row.comment_id) 
        if old is not None and old[0] == reference_key(row): 
            assigned.append(old[1]) 
            counts["reference_kept"] += 1 
            continue 
            
        known = Counter(vocab[w] for w in row.tokens if w in vocab) 
                
        length = sum(known.values())
        if length < 2: 
            assigned.append(-1) 
            counts["insufficient_known_tokens"] += 1 
            continue 
            
        lp = np.log(m[active] + alpha) 
        for word, count in known.items(): 
            for j in range(count): 
                lp += np.log(nkw[active, word] + beta + j) 
        for j in range(length): 
            lp -= np.log(n[active] + len(words) * beta + j) 
        assigned.append(int(active[np.argmax(lp)])) 
        counts["inferred"] += 1 

    df["cluster"] = assigned 
    df["aspect"] = [labels[k] if k != -1 else "Khác" for k in assigned] 

    report = { 
        "model": folder.name, 
        "manifest_sha256": file_sha256(manifest_path), 
        "input_rows": input_rows, 
        "output_rows": len(df), 
        "reference_kept": counts["reference_kept"], 
        "inferred": counts["inferred"], 
        "insufficient_known_tokens": counts["insufficient_known_tokens"], 

        "method": "frozen_dmm_predictive_argmax_with_training_reference", 
        "note": "cluster=-1: ít hơn 2 token trong từ điển; aspect=Khác; vẫn chạy sentiment.", 
        } 

    output.parent.mkdir(parents=True, exist_ok=True) 
    with tempfile.TemporaryDirectory(prefix=".topics-", dir=output.parent) as work: 
            snapshot = Path(work) / "result" 
            snapshot.mkdir() 
            df.to_parquet(snapshot / "comments_with_aspects.parquet", index=False) 
            (snapshot / "topic_prediction_summary.json").write_text( 
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8" 
            ) 
            snapshot.rename(output) 
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

def main(): 
    parser = argparse.ArgumentParser(description=__doc__) 
    parser.add_argument("seed", nargs="?", type=int, default=42) 
    parser.add_argument("--mode", choices=["train", "predict", "seal", "check"], default="train") 
    parser.add_argument("--model-dir", type=Path) 
    parser.add_argument("--input", type=Path, default=Path("/opt/data/silver/comments_clean")) 
    parser.add_argument("--output-dir", type=Path) 
    parser.add_argument("--iterations", type=int, default=30) 
    args = parser.parse_args() 
    if args.mode != "train": 
        if args.model_dir is None: 
            parser.error("Cần --model-dir khi predict/seal/check.") 
        run_fixed_model(args, parser) 
        return 
    if args.output_dir is None: 
        args.output_dir = Path("/opt/data/gold")
        
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

    # Silver của pipeline chuẩn hóa văn bản thành text_clean/text_raw; bản
    # GSDMM cũ lại yêu cầu cột text. Tạo alias tương thích, không ghi ngược
    # hoặc sửa file Silver nguồn.
    if "text" not in raw.columns:
        if "text_clean" in raw.columns:
            raw["text"] = raw["text_clean"]
        elif "text_raw" in raw.columns:
            raw["text"] = raw["text_raw"]

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