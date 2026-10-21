# Bộ ghép rút gọn: 73.247 bình luận

[Tải ZIP đầy đủ (khoảng 19 MB)](./ytb-mixed-comments-73247.zip?raw=true)

Nếu mở trang ZIP, chọn **Download raw file**. Giải nén được thư mục `mixed_comments_73247/`.

[Xem trước 100 dòng](./preview_100.csv).

- Lấy ngẫu nhiên **73.247 dòng** từ bản 75.000 dòng, bỏ **1.753 dòng**, không lặp; seed 42.
- Đúng **18 cột**, cùng schema với Core, không thêm nhãn hoặc cờ.
- Giữ nguyên nội dung, ID, ngày đăng, ngày cập nhật, processed_at và mọi giá trị của dòng được giữ.
- Ngày đăng UTC: **29/08/2026–21/10/2026**, đủ 54 ngày; không dàn lại ngày khi giảm số dòng.
- Gồm **48,809 dòng thực tế + 24,438 dòng giả lập theo mẫu**; không phải toàn bộ dữ liệu thật.
- ZIP gồm `comments.parquet`, `comments.csv`, bản xem trước, hướng dẫn và báo cáo.
- Bản 75.000 dòng, dữ liệu gốc và curate.py được giữ nguyên.

Phần thật lấy từ Core chưa lọc lại độ liên quan video với nhạc AI. Phần giả lập và ngày mô phỏng không phải bằng chứng về cảm xúc hay xu hướng người xem thật. Việc lấy mẫu có thể khiến reply không kèm cha.

Đã kiểm tra số dòng CSV/Parquet, 18 cột, ID duy nhất, round-trip Parquet, giữ đủ ngày và hash nguồn không đổi.

## Chạy GSDMM

Đặt thư mục giải nén vào `data/mixed_comments_73247/` trong repo, chạy từ Windows PowerShell ở thư mục repo:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/mixed_comments_73247/comments.parquet --output-dir /project/data/business/topics/mixed_73247_seed_42 --seed 42
```

Mô hình có các bước loại dòng nên số train sẽ nhỏ hơn 73.247. Chọn output mới nếu kết quả đã tồn tại.

SHA256 của ZIP:

```text
e3bb508f0b03d08694a410321e7c29baa587ee42f1b465e7d2314e1812b86af7
```
