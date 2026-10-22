import pytest

from ytb.relevance import is_ai_music


@pytest.mark.parametrize("title", [
    "Ai là người đang hát??? #meoxanh #ohnoteam #cosplay #halloween",
    "Ai mới là người hát - Cấp độ siêu khó",
    "Cô bị ép cưới một ngư dân nghèo, ai ngờ anh là đại gia giàu nhất biển",
    "a New RTS Game with deep AI cover mechanics | Defilade",
    "2 Ngày 1 Đêm Mùa 5 | Tập 105",
    "Ca Sĩ Mặt Nạ - The Masked Singer Vietnam 2026 Tập 3",
    "Ai nghe nhạc này chưa? Ai cover vậy?",
])
def test_loai_video_khong_lien_quan(title):
    assert not is_ai_music(title, "", [], None)


@pytest.mark.parametrize("title", [
    "Suno AI - cách tạo bài hát bằng AI",
    "Bài hát do AI sáng tác về mẹ",
    "AI Cover Hãy trao cho anh | Giọng AI",
    "Nhạc AI Việt Nam hay nhất 2026",
    "Giọng AI hát cover Sơn Tùng",
])
def test_nhan_video_nhac_ai(title):
    assert is_ai_music(title, "", [], None)


def test_nhan_qua_tags_va_mo_ta():
    assert is_ai_music("Hãy trao cho anh", "", ["ai cover", "remix"], None)
    assert is_ai_music("Bài hát mới", "Nhạc làm bằng Suno", [], None)
    assert not is_ai_music("Chuyện đời", "Nhạc làm bằng Suno", [], None)   # tiêu đề không nói về nhạc