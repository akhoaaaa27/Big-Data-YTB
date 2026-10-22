# Kết quả kiểm tra khi đóng gói — 22/10/2026

- `python -m pytest -q`: **14 passed**. Chạy trên môi trường Python 3.12.14, NumPy 2.5.4, pandas 2.2.3, PyArrow 25.0.1. Image được cấu hình Python 3.10 và NumPy 2.2.6 tương thích Python 3.10; chưa xác nhận hoàn tất build image ở môi trường kiểm tra này.
- Kiểm tra tích hợp dùng 2 file Parquet nhỏ tự tạo cho test, chạy GSDMM thật 3 vòng, gán một tên cụm cho mỗi dòng, rồi kiểm tra đầu ra sentiment bằng predictor giả định. Xác minh Silver không thay đổi, `comment_id` không nhân đôi và `aspect` là một chuỗi duy nhất. Dữ liệu test không phải dữ liệu phân tích.
- Đối chiếu bảng tên cụm tham khảo: tổng 34.923 dòng, `Khác` 12.953 dòng, đúng thống kê lần chạy người dùng đã cung cấp.
- Kiểm tra ánh xạ nhãn từ cấu hình mô hình, giữ từ phủ định/emoji, giữ dòng không còn chữ với nhãn trống, từ chối xác suất không hợp lệ và không ghi đè kết quả cũ.
- `docker compose config --quiet`: đạt.
- Hai Dockerfile đã được đưa vào build thật. Môi trường kiểm tra tải được base image, nhưng container build không phân giải được tên miền kho thư viện (`Temporary failure in name resolution`), nên chưa hoàn tất cài dependency hoặc chạy suy luận VisoBERT trong Docker tại đây.

Các test có predictor giả chỉ xác minh xử lý dữ liệu và đầu ra, không kiểm chứng độ chính xác sentiment. Trên máy sử dụng, chạy thử 100 dòng theo README trước khi chạy toàn bộ. Không có kết quả accuracy hoặc F1 độc lập cho bộ bình luận này.
