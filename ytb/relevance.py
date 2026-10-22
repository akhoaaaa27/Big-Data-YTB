"""Quyết định một video có thật sự nói về nhạc AI hay không.

Vấn đề: "ai" trong tiếng Việt là đại từ ("Ai là người đang hát?", "ai ngờ...").
Cách xử lý: chữ AI phải viết HOA và đứng cạnh một từ về nhạc/giọng/cover mới tính.
Các từ không nhập nhằng (suno, udio, rvc) tính luôn. Chỉ là luật đơn giản, cần kiểm bằng mẫu tay.
"""

from __future__ import annotations

import re

# Từ khóa không nhập nhằng
_UNAMBIG = re.compile(r"\b(?:suno|udio|rvc|so-?vits(?:-?svc)?)\b", re.I)

# "AI" phải viết hoa (phần (?i:...) chỉ áp dụng cho từ đi kèm)
_AI_PHRASE = re.compile(
    r"(?i:nhạc|bài hát|ca khúc|giọng hát|giọng|ca sĩ|cover|remix|song|music|voice|singer|vocals?)"
    r"\s+(?:(?i:do|bằng|by)\s+)?AI\b"
    r"|\bAI[\s\-]+(?i:cover|music|song|voice|singer|vocals?|generated|nhạc|bài hát|ca sĩ|giọng|hát|sáng tác|tạo)"
)

# Thẻ (tags) do người đăng chọn nên không nhập nhằng như câu văn: so khớp thường được
_TAG_TERMS = re.compile(
    r"\bsuno\b|\budio\b|\brvc\b|so-?vits|aicover|aimusic"
    r"|\bai[\s\-]?(?:cover|music|song|voice|singer|vocals?|generated)\b"
    r"|(?:nhạc|bài hát|ca sĩ|giọng)\s+ai\b"
)

_MUSIC = re.compile(r"nhạc|bài hát|ca khúc|\bhát\b|\bcover\b|\bsong\b|\bmusic\b|giọng|\bremix\b|\bmv\b|\blyrics\b|ca sĩ", re.I)
_GAME = re.compile(r"\b(?:rts|gameplay|steam)\b", re.I)


def _is_shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def classify(title: str | None, description: str | None, tags: list[str] | None,
             synthetic: bool | None = None) -> tuple[bool, str]:
    """Trả về (có phải nhạc AI không, lý do) để dễ kiểm tra."""
    title, desc = title or "", description or ""
    tags_l = " ".join(tags or []).lower()

    if _GAME.search(title) and not _UNAMBIG.search(title + " " + tags_l):
        return False, "game"
    if _UNAMBIG.search(title):
        return True, "title:unambiguous"
    # Tiêu đề viết hoa toàn bộ thì "AI" có thể là đại từ viết hoa, bỏ qua tín hiệu này
    if not _is_shouting(title) and _AI_PHRASE.search(title):
        return True, "title:ai_phrase"
    if _TAG_TERMS.search(tags_l):
        return True, "tags"
    if _MUSIC.search(title + " " + tags_l):
        if _UNAMBIG.search(desc) or _AI_PHRASE.search(desc):
            return True, "description"
        if synthetic:
            return True, "synthetic_flag"
    return False, "no_signal"


def is_ai_music(title, description=None, tags=None, synthetic=None) -> bool:
    return classify(title, description, tags, synthetic)[0]