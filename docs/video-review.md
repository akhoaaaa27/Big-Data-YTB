# Lọc nhạc AI tự động

Không cần mở CSV hay điền nhãn video bằng tay. Từ khóa tìm kiếm, tên kênh và cờ
`contains_synthetic_media` không được dùng riêng lẻ để kết luận âm nhạc do AI tạo.

Chạy tại thư mục repository:

```bash
YTB_DATA_DIR="$PWD/data" python -m ytb.curate
```

Hoặc trong Docker của project:

```bash
docker compose --profile debug run --rm airflow-cli python -m ytb.curate
```

Bộ lọc đọc metadata của `data/core/videos`, tự tạo ba nhóm:

- `include`: có bằng chứng cụ thể về nhạc/giọng hát/cover AI trong tiêu đề hoặc
  mô tả, hoặc tiêu đề nêu công cụ tạo nhạc Suno/Udio.
- `exclude`: tiêu đề chỉ rõ nội dung hướng dẫn, game, review phim hoặc test loa.
- `uncertain`: thiếu bằng chứng, chỉ có tags/cờ synthetic, chỉ thấy hình ảnh AI,
  hoặc có thông tin phủ định việc dùng AI. Nhóm này tự động được tách khỏi tập
  phân tích; không cần người dùng duyệt để hoàn thành lệnh.

Ví dụ “AI cover mechanics” trong game bị loại, “Ai là người đang hát?” không
được coi là giọng hát AI. “Nhạc và ca sĩ: Suno AI” là bằng chứng về âm thanh.
Quy tắc được chuẩn hóa Unicode và cách viết `A.I` trước khi so khớp.

Kết quả:

```text
data/curated/videos/           # video được tự động giữ
data/curated/comments/         # bình luận chỉ thuộc các video được giữ
data/curated/video_snapshots/  # snapshot cùng tập video
data/curated/classification.csv # phân loại, lý do và bằng chứng cho mọi video
data/curated/uncertain.csv     # các trường hợp thiếu bằng chứng
data/curated/review.json       # phiên bản bộ lọc, thời gian và số dòng giữ/loại
```

`data/review/videos.csv` cũng được cập nhật thành **báo cáo tự động**, không còn
là biểu mẫu cần điền. Lệnh `apply` cũ là bí danh của chế độ tự động; không đọc
quyết định gán tay từ CSV. `export` chỉ xuất báo cáo phân loại, không xuất Parquet.

Bộ lọc dựa trên khai báo trong metadata, chưa kiểm chứng âm thanh thực tế, nên
không phải nhãn chuẩn hay cam kết chính xác 100%. Video không khai báo AI có
thể bị tách sang uncertain dù thực sự là nhạc AI. Bộ lọc không tự suy ra ngôn ngữ
hay quốc tịch người nghe từ `regionCode`/`relevanceLanguage`.

Core và Staging gốc được giữ nguyên. Đầu ra mới được ghi hoàn chỉnh vào thư mục
tạm rồi thay lớp curated; nếu không có video phù hợp thì xuất bảng rỗng để không
vô tình dùng lại kết quả cũ. Còn `curated.old` sau sự cố thì cần khôi phục trước
khi chạy tiếp. Bộ lọc không hỗ trợ hai tiến trình ghi curated đồng thời.

DAG tự chạy `curate_videos` sau `spark_core`. Bước lấy mẫu gán nhãn **cảm xúc**
đọc curated và kiểm tra báo cáo đi kèm; việc gán cảm xúc là bước nghiên cứu khác,
không phải yêu cầu gán tay nhãn video AI. Các phiếu cảm xúc đã xuất trước đây
không tự thay đổi; cần tạo bộ mới để sử dụng tập video đã lọc.
