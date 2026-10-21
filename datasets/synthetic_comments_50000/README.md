# Bộ GIẢ LẬP: 50.000 bình luận nhạc AI tiếng Việt

**Toàn bộ dữ liệu được sinh từ mẫu, không phải bình luận YouTube thật.**
Nhãn sentiment và khía cạnh được gán khi sinh, chưa được người đánh giá độc lập.
Dùng kiểm thử pipeline, schema và hiệu năng; không dùng làm bằng chứng về
cảm xúc người xem thật hoặc độ chính xác mô hình trên dữ liệu thực tế.

## Tải xuống

[Tải ZIP đầy đủ (khoảng 10 MB)](./ytb-synthetic-comments-50000.zip?raw=true)

Bấm liên kết trên; hoặc mở file ZIP trong folder này rồi chọn **Download raw file**.
Giải nén sẽ được thư mục `synthetic_comments_50000/`.
Không cần tải toàn bộ repository.

[Xem trước 100 bình luận](./preview_100.csv).

## Nội dung ZIP

- `comments.csv`: đủ 50.000 dòng dữ liệu và một dòng tiêu đề, UTF-8 BOM để mở bằng Excel.
- `comments/part-00000.parquet` đến `part-00003.parquet`: mỗi file 12.500 dòng, cùng schema.
- `videos/part-00000.parquet`: 500 video giả lập để nối theo `video_id`.
- `summary.json`, `validation_report.json`: thông tin và kiểm tra dữ liệu.
- `README.md`, `preview_100.csv`: hướng dẫn và bản xem nhanh.

## Đặc điểm

- Đủ 18 cột Core yêu cầu, thêm `is_synthetic`, `synthetic_aspect`, `synthetic_sentiment`.
- `source=synthetic`; mọi ID bắt đầu bằng `SYN_`, không giả danh ID hoặc nguồn API thật.
- Ngày đăng UTC từ **01/01/2025 đến 20/10/2026**, có dữ liệu ở cả 658 ngày.
- **35.000 tích cực (70%), 10.000 trung lập (20%), 5.000 tiêu cực (10%)**.
- 20 khía cạnh, mỗi khía cạnh 2.500 dòng; một nhãn chính mỗi bình luận.
- 40.000 bình luận gốc và 10.000 reply; ID cha, thread và số reply khớp nhau.
- 50.000 ID duy nhất và 50.000 `text_clean` khác nhau. Tuy nhiên câu được ghép
  từ các mẫu dùng lại, không phải 50.000 bình luận được viết độc lập.
- Làm sạch và tách từ bằng `ytb.text_clean` của repo; giữ emoji trong `text_clean`.
- Đã kiểm tra số dòng CSV/Parquet, schema, timestamp, quan hệ reply, số token,
  và chạy GSDMM một vòng trên mẫu 1.000 dòng (800 dòng không phải reply).

## Chạy thử GSDMM

Đặt thư mục đã giải nén vào `data/synthetic_comments_50000/` trong repo.
Trong Windows PowerShell ở thư mục repo, với image `gsdmm-local` đã có numpy và pyarrow:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/synthetic_comments_50000/comments --output-dir /project/data/business/topics/synthetic_seed_42 --seed 42
```

GSDMM sẽ bỏ reply, khử trùng và lọc từ hiếm; số train không mặc định bằng 50.000.
Chọn thư mục output mới nếu thư mục kết quả đã tồn tại. Giữ bộ giả lập tách biệt
với `data/core/`, không thay thế dữ liệu thu thập thật.

SHA256 của ZIP:

```text
3ab32a2bac62b3fc480f99dd136f0d3210770381ff9be9bce290dbe2cdcac715
```
