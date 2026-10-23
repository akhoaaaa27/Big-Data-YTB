import argparse
import os
import sys
from pathlib import Path

import re
import unicodedata

from pyspark.sql import SparkSession, functions as F, types as T


def _normalize_offtopic(value):
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(value or "").lower())
    text = unicodedata.normalize("NFD", text).replace("đ", "d")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


# Giữ phản hồi có ngữ cảnh âm nhạc; không dùng chữ "ai" đơn lẻ làm bằng chứng AI.
_AUDIO_CONTEXT = re.compile(
    r"\b(?:suno|udio|nhac|hat|giong|ai cover|ai music|ban phoi|phoi khi|hoa am|"
    r"bai hat|ca khuc|giai dieu|loi bai|loi hat|vocal|beat|instrumental|karaoke|"
    r"nhac nen|nhac phim|soundtrack|mv)\b"
)
_FILM = re.compile(r"\b(?:phim|thuyet minh)\b")
_ACTOR = re.compile(r"\b(?:dien vien|dao dien)\b")
_ACTING = re.compile(r"\b(?:vai dien|dong vai|nu chinh|nam chinh|kich ban|canh quay)\b")
_SHOW = re.compile(
    r"\b(?:khach moi|dan cast|cast|doi hinh|cris phan|cris|nicky|"
    r"truong giang|lam vy da|le duong bao lam|ong lam)\b"
)
_SHOW_CHAT = re.compile(r"\b(?:cuoi|hai qua|hai ghe|bao hai|tau hai|hai huoc|man mieng|rang tam)\b")
_MASCOT = re.compile(r"\b(?:gauci|achu|minzy|lion|khi xu|heo ma hong|bo cong|xu mi|cao boi)\b")
_GUESS = re.compile(r"\b(?:doan|la ai|chac la|co the la|nghi la|lo dien|chon so)\b")
_GAME = re.compile(r"\b(?:minecraft|roblox|free fire|lien quan mobile|pubg|genshin|gameplay)\b")
_SPORT = re.compile(r"\b(?:ban thang|penalty|phat den|cau thu|ngoai hang anh|world cup)\b")


def off_topic_reason(value):
    text = _normalize_offtopic(value)
    if not text or _AUDIO_CONTEXT.search(text):
        return None
    if _FILM.search(text) or (_ACTOR.search(text) and _ACTING.search(text)):
        return "phim_dien_xuat"
    if _SHOW.search(text) and _SHOW_CHAT.search(text):
        return "show_hai_tro_chuyen"
    if _MASCOT.search(text) and _GUESS.search(text):
        return "doan_mascot"
    if _GAME.search(text):
        return "tro_choi"
    if _SPORT.search(text):
        return "the_thao"
    return None

def run_clean(input_path, output_dir, audit_dir, spark=None): 
    input_path, output_dir, audit_dir = map(Path, (input_path, output_dir, audit_dir)) 
    for destination in (output_dir, audit_dir): 
        src, dst = input_path.resolve(), destination.resolve() 
        if src == dst or src in dst.parents or dst in src.parents: 
            raise ValueError("Đầu vào và đầu ra phải tách biệt.") 
    out, audit = output_dir.resolve(), audit_dir.resolve() 
    if out == audit or out in audit.parents or audit in out.parents: 
        raise ValueError("Silver và báo cáo loại phải tách biệt.") 
    own_spark = spark is None 
    if own_spark: 
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable) 
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable) 
        spark = ( 
                SparkSession.builder.appName("ai-music-comments") 
                .master(os.getenv("YTB_SPARK_MASTER", "local[2]")) 
                .config("spark.sql.session.timeZone", "UTC") 
                .config("spark.ui.enabled", "false") 
                .getOrCreate() 
                ) 
        
    tagged = None 
    try: 
        raw = spark.read.parquet(str(input_path)) 
        print("Cột gốc:", raw.columns) 
        print("Số dòng ban đầu:", raw.count()) 
        df = ( 
                raw.withColumnRenamed("text_raw", "text") 
                .withColumnRenamed("like_count", "likes") 
                .withColumnRenamed("author_channel_id", "author") ) 
        if "published" not in df.columns:
            if "published_date" in df.columns:
                df = df.withColumnRenamed("published_date", "published")
            elif "published_at" in df.columns:
                df = df.withColumnRenamed("published_at", "published")

        required = {
                "comment_id",
                "video_id",
                "author",
                "text",
                "likes",
                "published",
                "is_reply",
                "tokens",
                }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Thiếu cột đầu vào: {sorted(missing)}")

        df = df.select(*sorted(required)) 
        df = ( 
                    df.withColumn("published", F.to_timestamp("published")) 
                    .withColumn("likes", F.col("likes").cast("int")) 
                    .fillna({"likes": 0}) 
                    .filter(~F.coalesce(F.col("is_reply"), F.lit(False))) 
                    .filter(F.col("text").isNotNull() & F.col("tokens").isNotNull()) 
                    .filter(F.col("comment_id").isNotNull() & F.col("video_id").isNotNull()) 
                    .filter((F.length(F.trim("comment_id")) > 0) & (F.length(F.trim("video_id")) > 0)) 
                    .dropDuplicates(["comment_id"]) 
                    .drop("is_reply") ) 
                
        print("Sau khi bỏ reply/rỗng/trùng ID:", df.count()) 
                
        df = ( 
                    df.withColumn("text_clean", F.lower("text")) 
                    .withColumn("text_clean", F.regexp_replace("text_clean", r"https?://\S+", " ")) 
                    .withColumn("text_clean", F.regexp_replace("text_clean", r"(.)\1{2,}", "$1")) 
                    .withColumn("text_clean", F.regexp_replace("text_clean", r"\s+", " ")) 
                    .withColumn("text_clean", F.trim("text_clean")) ) 
                
        vi_chars = "[ăơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệịỉĩọỏốồổỗộớờởỡợụủũứừửữựỳỵỷỹ]" 
        vi_words = ( 
                r"(?<!\p{L})(không|của|và|là|này|rất|quá|bài|nghe|giọng|nhạc|được|" 
                r"nhưng|chị|mình|bạn)(?!\p{L})" ) 
                
        math = ( 
                r"[²³⁴√∑∫≈≠≤≥]|\^\d|chu vi|diện tích|lập phương|bình phương|đẳng thức|" 
                r"phương trình|căn bậc|đạo hàm|tích phân|bài toán|giải toán" ) 
                
        df = ( 
                df.withColumn("is_vi", F.col("text_clean").rlike(vi_chars) | F.col("text_clean").rlike(vi_words)) 
                .withColumn("is_toan", F.col("text_clean").rlike(math)) )

        print("Không phải tiếng Việt:", df.filter(~F.col("is_vi")).count()) 
                
        print("Toán:", df.filter(F.col("is_vi") & F.col("is_toan")).count()) 
        df = df.filter( 
                    F.col("is_vi") & ~F.col("is_toan") 
                    & (F.size(F.split("text_clean", " ")) >= 3) ) 
                
        print("Sau lọc ngôn ngữ/toán/độ dài:", df.count()) 
                
        df = df.withColumn("n_words", F.size(F.split("text_clean", " "))) 
        short = df.filter("n_words < 8") 
        long = df.filter("n_words >= 8").dropDuplicates(["text_clean"]) 
        df = short.unionByName(long).drop("n_words") 
        tagged = df.withColumn( 
                    "_off_topic_reason", F.udf(off_topic_reason, T.StringType())("text") 
                    ).cache() 
                
        before = tagged.count() 
        removed = tagged.filter(F.col("_off_topic_reason").isNotNull()) 
        n_removed = removed.count() 
                
        print("Trước lọc lạc đề:", before) 
        print("Loại do lạc đề:", n_removed) 
        print("Giữ lại:", before - n_removed) 
                
        if before - n_removed <= 0: 
            raise ValueError("Không còn bình luận; không xuất Silver rỗng.") 
        removed.groupBy("_off_topic_reason").count().show(truncate=False) 
        removed.select( 
                    "comment_id", "video_id", 
                    "text", "_off_topic_reason" ).write.mode("overwrite").parquet(str(audit_dir)) 
                
        kept = tagged.filter(
                    F.col("_off_topic_reason").isNull()).select( 
                        "comment_id", "video_id", "author", "text", 
                        "likes", "published", "tokens" 
                        ) 
                
        kept.coalesce(1).write.mode("overwrite").parquet(str(output_dir)) 
                
        return output_dir 
    finally: 
        if tagged is not None: 
            tagged.unpersist() 
        if own_spark: 
            spark.stop() 
                
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input", required=True) 
    parser.add_argument("--output", required=True) 
    parser.add_argument("--audit", required=True) 
    args = parser.parse_args() 
    run_clean(args.input, args.output, args.audit)
