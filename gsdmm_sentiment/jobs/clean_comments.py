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

spark = SparkSession.builder.appName("ai-music-comments").getOrCreate()

raw = spark.read.parquet("/opt/data/raw/test.parquet")
print("Cột gốc:", raw.columns)
print("Số dòng ban đầu:", raw.count())

# 0. Đổi tên cột về tên chuẩn, giữ nguyên tokens gốc
#    (dùng published_date vì published_at bị lỗi kiểu INT64 khi Spark đọc)
df = (raw.withColumnRenamed("text_raw", "text")
         .withColumnRenamed("like_count", "likes")
         .withColumnRenamed("published_date", "published")
         .withColumnRenamed("author_channel_id", "author")
         .select("comment_id", "video_id", "author", "text", "likes", "published",
                 "is_reply", "source", "tokens"))

# 1. Kiểu dữ liệu; tạo cờ dữ liệu mô phỏng rồi bỏ cột source
df = (df.withColumn("published", F.to_timestamp("published"))
        .withColumn("likes", F.col("likes").cast("int"))
        .fillna({"likes": 0})
        .withColumn("is_synth", F.col("source") == "synthetic")
        .drop("source"))

# 2. Bỏ bình luận trả lời, rỗng, trùng id, thiếu tokens
df = (df.filter(~F.coalesce(F.col("is_reply"), F.lit(False)))
        .filter(F.col("text").isNotNull() & F.col("tokens").isNotNull())
        .dropDuplicates(["comment_id"])
        .drop("is_reply"))
print("Sau khi bỏ reply, rỗng, trùng id:", df.count())
df.groupBy("is_synth").count().show()

# 3. Chuẩn hóa văn bản (dùng để lọc ngôn ngữ và cho sentiment)
df = (df.withColumn("text_clean", F.lower(F.col("text")))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"https?://\S+", " "))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"(.)\1{2,}", "$1"))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"\s+", " "))
        .withColumn("text_clean", F.trim("text_clean")))

# 4. Lọc tiếng nước ngoài, toán
VI_ONLY = "[ăơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệịỉĩọỏốồổỗộớờởỡợụủũứừửữựỳỵỷỹ]"
VI_WORDS = (r"(?<!\p{L})(không|của|và|là|này|rất|quá|bài|nghe|giọng|nhạc|được|"
            r"nhưng|chị|mình|bạn)(?!\p{L})")
TOAN = (r"[²³⁴√∑∫≈≠≤≥]|\^\d|chu vi|diện tích|lập phương|bình phương|đẳng thức|"
        r"phương trình|căn bậc|đạo hàm|tích phân|bài toán|giải toán")

df = (df.withColumn("is_vi", F.col("text_clean").rlike(VI_ONLY) | F.col("text_clean").rlike(VI_WORDS))
        .withColumn("is_toan", F.col("text_clean").rlike(TOAN)))

print("Không phải tiếng Việt:", df.filter(~F.col("is_vi")).count())
print("Toán:", df.filter(F.col("is_vi") & F.col("is_toan")).count())

df_vi = df.filter(F.col("is_vi") & ~F.col("is_toan")
                  & (F.size(F.split("text_clean", " ")) >= 3))
print("Sau lọc ngôn ngữ/toán/độ dài:", df_vi.count())

# 5. Bỏ trùng bình luận dài (lời bài hát dán lại), giữ bình luận ngắn lặp lại
df_vi = df_vi.withColumn("n_words", F.size(F.split("text_clean", " ")))
long_c = df_vi.filter("n_words >= 8").dropDuplicates(["text_clean"])
short_c = df_vi.filter("n_words < 8")
df_vi = short_c.unionByName(long_c).drop("n_words")

# 6. Lọc lạc đề theo cụm từ gợi ý từ GSDMM; giữ nguyên nội dung và tokens.
# Đây là heuristic giảm nhiễu, không xác minh video/âm thanh có phải nhạc AI.
classify_offtopic = F.udf(off_topic_reason, T.StringType())
tagged = df_vi.withColumn("_off_topic_reason", classify_offtopic(F.col("text"))).cache()
try:
    n_before = tagged.count()
    removed = tagged.filter(F.col("_off_topic_reason").isNotNull())
    n_removed = removed.count()
    print("Trước lọc lạc đề:", n_before)
    print("Loại do lạc đề:", n_removed)
    print("Giữ lại sau lọc lạc đề:", n_before - n_removed)
    removed.groupBy("_off_topic_reason").count().orderBy(F.desc("count")).show(truncate=False)
    # Báo cáo riêng để kiểm tra dòng bị loại; không thêm cột vào Silver chính.
    removed.select("comment_id", "video_id", "text", "_off_topic_reason").write.mode("overwrite").parquet(
        "/opt/data/gold/comments_removed_offtopic"
    )
    df_vi = tagged.filter(F.col("_off_topic_reason").isNull()).drop("_off_topic_reason")
    df_vi = df_vi.drop("text_clean", "is_vi", "is_toan", "is_synth")
    print("Số dòng sau khi lọc:", n_before - n_removed)
    df_vi.coalesce(1).write.mode("overwrite").parquet(
        "/opt/data/silver/comments_clean"
    )
finally:
    tagged.unpersist()
    spark.stop()

