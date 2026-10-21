"""Làm sạch và tách từ bình luận tiếng Việt (thuần Python, không cần Spark nên test được riêng).

Hai đầu ra cho mỗi bình luận:
  - text_clean: văn bản đã làm sạch, GIỮ emoji và dấu câu (dùng cho mô hình cảm xúc, vì emoji mang cảm xúc).
  - tokens: danh sách từ đã tách, bỏ emoji, dấu câu, từ dừng (dùng cho GSDMM tìm chủ đề).
"""

from __future__ import annotations

import html
import re
import unicodedata

# Bình luận có ít hơn số từ này thì không đưa vào mô hình chủ đề (quá ngắn để gom cụm).
# Con số tạm, sẽ chỉnh ở bước 3b sau khi xem kết quả thực tế.
MIN_TOKENS = 3

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@[\w.\-]+")
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")      # 3:45 hoặc 1:02:10 (mốc thời gian trong video)
# Chữ kéo dài: "quáaaa", "đẹpppp", "ăăăn". Làm trên dạng NFD (dấu tách rời) để "á"+"aaa" cũng gom được.
_REPEAT_SAME = re.compile(r"(([^\W\d_])[\u0300-\u036f]*)\1{2,}")
_REPEAT_AFTER_MARK = re.compile(r"(([^\W\d_])[\u0300-\u036f]+)\2{2,}")
_NOT_WORD = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")

# Từ dừng tiếng Việt, chỉ dùng cho tách chủ đề (KHÔNG áp dụng cho text_clean).
# KHÔNG đưa "ai" (chính là AI, chủ đề của đồ án) và "hay" (khen hay) vào đây.
# Cố ý giữ danh sách ngắn; có thể chỉnh sau khi xem từ nổi bật của từng cụm.
STOPWORDS = frozenset("""
và là của thì mà có cũng này đó ấy các những một cho với được bị để rồi nhé nha ạ à ơi thế vậy nhưng nên
như khi đã đang sẽ ở trong ra lên mình tôi em anh chị gì sao đây kia thôi luôn lại còn cứ chỉ rất quá
không chưa nó họ ta bạn mọi mấy hơn nữa đều vẫn cả từ về tới vì nếu hoặc hả ừ ờ dạ vâng ok ko tui
""".split())

# Một phần bình luận là tiếng Anh (khán giả nước ngoài). Bỏ từ dừng tiếng Anh để chúng không chiếm chỗ
# trong danh sách từ nổi bật của cụm. Các bình luận tiếng Anh vẫn còn trong dữ liệu.
# Cố ý KHÔNG đưa vào các từ trùng với tiếng Việt không dấu: do (do AI hát), to (to quá), so, an, me, my (Mỹ), be (bé), can.
STOPWORDS_EN = frozenset("""
the and you it this is of in for that we are was with on have but your what just they not at as
he she if all from by will or one how its im dont very too
""".split())
STOPWORDS = STOPWORDS | STOPWORDS_EN

_tokenizer = None


def _vi_tokenize(text: str) -> str:
    """Nạp pyvi một lần cho mỗi tiến trình (nạp mô hình khá tốn thời gian)."""
    global _tokenizer
    if _tokenizer is None:
        from pyvi import ViTokenizer
        _tokenizer = ViTokenizer
    return _tokenizer.tokenize(text)


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    t = html.unescape(raw)
    t = _TAG.sub(" ", t)
    t = unicodedata.normalize("NFC", t).lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _TIMESTAMP.sub(" ", t)
    t = unicodedata.normalize("NFD", t)
    t = _REPEAT_AFTER_MARK.sub(r"\1", t)   # phải làm trước: "á"+"aaa" -> "á"
    t = _REPEAT_SAME.sub(r"\1", t)
    t = unicodedata.normalize("NFC", t)
    return _SPACES.sub(" ", t).strip()


def tokenize(text_clean: str) -> list[str]:
    if not text_clean:
        return []
    words_only = _SPACES.sub(" ", _NOT_WORD.sub(" ", text_clean)).strip()
    if not words_only:
        return []
    out = []
    for tok in _vi_tokenize(words_only).split():
        if tok in STOPWORDS or tok.isdigit() or len(tok) < 2:
            continue
        out.append(tok)
    return out


def process(raw: str | None) -> tuple[str, list[str]]:
    clean = clean_text(raw)
    return clean, tokenize(clean)