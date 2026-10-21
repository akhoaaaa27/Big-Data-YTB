# Lọc đúng đối tượng nhạc AI

Tìm kiếm YouTube chỉ tạo danh sách ứng viên. `keyword` là từ khóa đã dùng để tìm,
không phải nhãn nội dung. Cờ `contains_synthetic_media` cũng có thể nói về hình ảnh
hoặc nội dung tổng hợp khác, không đủ để kết luận âm nhạc do AI tạo.

`data/core/` giữ dữ liệu ứng viên sau khử trùng/làm sạch văn bản. Lớp dùng cho
gán nhãn và phân tích là `data/curated/`: chỉ chứa video đã được duyệt, cùng
bình luận và snapshot liên quan. Các file Core và Staging không bị xóa/sửa.

## Với dữ liệu đã có

Chạy tại thư mục repository (trong Docker bỏ tiền tố `YTB_DATA_DIR=...` vì
container đã gắn dữ liệu vào `/opt/airflow/data`):

```bash
YTB_DATA_DIR="$PWD/data" python -m ytb.curate export
```

Mở `data/review/videos.csv` bằng Excel/Sheets. Cột `suggestion` và `reason` chỉ
là gợi ý ưu tiên kiểm tra, không tự động duyệt. Danh sách được sắp theo
`n_comments` giảm dần; có thể ưu tiên video có bình luận để làm sạch bộ mẫu đang có.
Điền cột `decision`:

- `include`: xác nhận đây là tác phẩm âm nhạc được AI tạo/hát/cover, phù hợp
  phạm vi nghiên cứu. Kiểm tra video, mô tả và nguồn công bố khi cần.
- `exclude`: video thường, hướng dẫn công cụ, review/tin tức, hoặc ngoài phạm vi.
- Để trống: chưa xác minh được; tạm thời không đưa vào dữ liệu phân tích.

Chữ “ai” trong tiếng Việt, hashtag, tên kênh và kết quả tìm kiếm không đủ để
xác nhận. Nếu chỉ nghiên cứu nhạc Việt, loại cả tác phẩm AI ngoài phạm vi ngôn ngữ.
Ghi căn cứ hoặc trường hợp khó vào cột `note`; không sửa `video_id`.

```bash
YTB_DATA_DIR="$PWD/data" python -m ytb.curate apply
YTB_DATA_DIR="$PWD/data" python -m ytb.make_label_sample
```

Lệnh `apply` lọc đồng bộ ba bảng theo danh sách `video_id` được duyệt và xuất
thống kê giữ/loại vào `data/curated/review.json`. Nó thay lớp curated sau khi ghi
xong, giữ nguyên schema và không thay Core. Chưa có video được duyệt thì lệnh
báo lỗi và giữ nguyên curated cũ; không dùng curated cũ sau khi sửa quyết định mà
chưa chạy lại `apply`. Kiểm tra timestamp và thống kê của lần xuất thành công.

## Những lần thu thập sau

DAG có bước `curate_videos` sau `spark_core`. Bước này cập nhật danh sách duyệt
(giữ quyết định và ghi chú cũ), sau đó xuất curated. Video mới chưa duyệt được
tạm loại khỏi curated. Nếu chưa có quyết định `include`, bước này thất bại để
không chạy tiếp phân tích trên dữ liệu chưa duyệt. Các bước Business/Mart khi
triển khai cần đọc `data/curated/`, không đọc trực tiếp Core.

Các phiếu gán nhãn đã xuất trước đây không tự cập nhật. Giữ lại để đối chiếu;
tạo bộ phiếu mới sau khi duyệt, tránh ghi đè nhãn đã chấm bằng `--force`.
