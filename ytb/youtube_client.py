"""Gọi YouTube Data API: đếm quota, gom các loại lỗi về vài lỗi dễ xử lý."""

from __future__ import annotations

import json
import os

from googleapiclient.errors import HttpError


class QuotaExhausted(Exception):
    """YouTube báo hết quota trong ngày. Dừng sớm, lần chạy sau sẽ làm tiếp."""


class BudgetExceeded(Exception):
    """Đã dùng hết ngân sách của lần chạy này (do mình tự đặt, chưa phải hết quota thật)."""


class CommentsUnavailable(Exception):
    """Video tắt bình luận, bị xóa hoặc không truy cập được: khỏi quét lại."""


class Budget:
    """Đếm số lệnh gọi API trong một task, để không dùng quá ngân sách."""

    def __init__(self, units: int, search_calls: int = 0):
        self.units_limit = units
        self.search_limit = search_calls
        self.units_used = 0
        self.search_used = 0

    def charge(self, kind: str) -> None:
        if kind == "search":
            if self.search_used >= self.search_limit:
                raise BudgetExceeded("hết ngân sách lượt search")
            self.search_used += 1
        else:
            if self.units_used >= self.units_limit:
                raise BudgetExceeded("hết ngân sách quota")
            self.units_used += 1


def build_client(api_key: str | None = None):
    from googleapiclient.discovery import build

    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu YOUTUBE_API_KEY. Thêm dòng YOUTUBE_API_KEY=... vào file .env rồi chạy lại."
        )
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _reason(error: HttpError) -> str:
    """Lấy mã lý do lỗi (ví dụ quotaExceeded, commentsDisabled) từ phản hồi của API."""
    try:
        data = json.loads(error.content.decode("utf-8"))
        return data["error"]["errors"][0].get("reason", "")
    except Exception:
        return str(error)


def call(request, budget: Budget, kind: str = "read"):
    """Chạy một request: tính ngân sách, thử lại khi lỗi tạm thời, đổi lỗi API thành lỗi dễ xử lý."""
    budget.charge(kind)
    try:
        return request.execute(num_retries=3)
    except HttpError as e:
        reason = _reason(e)
        if "quotaExceeded" in reason or "dailyLimitExceeded" in reason:
            raise QuotaExhausted(reason) from e
        if any(k in reason for k in ("commentsDisabled", "videoNotFound", "forbidden")):
            raise CommentsUnavailable(reason) from e
        raise
