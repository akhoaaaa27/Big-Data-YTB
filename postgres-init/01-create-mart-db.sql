-- Chạy một lần khi PostgreSQL khởi tạo lần đầu (volume còn trống).
-- Tạo database riêng cho bảng Mart, tách khỏi database metadata của Airflow.
CREATE DATABASE ytb_mart;
