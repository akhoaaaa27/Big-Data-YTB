# Bộ dữ liệu trộn 73.247 dòng v3 — ID mô phỏng theo định dạng YouTube

[Tải ZIP đầy đủ](./ytb-mixed-comments-73247-v3.zip?raw=true).
[Xem 100 dòng giả lập có ID mới](./synthetic_preview_100.csv).
[Xem trước bản trộn](./preview_100.csv).

Bản này lấy nguyên bộ v2 gồm **48.809 dòng thật và 24.438 dòng giả lập**,
chỉ đổi ID của phần giả lập. Đây vẫn là dữ liệu hỗn hợp, không phải toàn bộ
bình luận YouTube thật. Cột `source` giữ nguyên để có thể kiểm tra nguồn.
Các ID mới chỉ là chuỗi mô phỏng đúng dạng ký tự/độ dài thường gặp; không
khẳng định chúng thuộc tài khoản, bình luận hoặc video YouTube tồn tại.

## Những gì thay đổi

- `comment_id`: ID gốc dài 26 ký tự; ID reply gồm ID cha, dấu chấm và hậu tố,
  tổng cộng 49 ký tự trong bộ dữ liệu này.
- `video_id`: 11 ký tự gồm chữ, số, gạch dưới hoặc gạch ngang.
- `author_channel_id`: `UC` và 22 ký tự.
- `thread_id` và `parent_id`: ánh xạ nhất quán với ID bình luận mới.

Không còn chuỗi `SYN` trong năm cột ID của các dòng giả lập. Mọi dòng thật
giữ nguyên toàn bộ 18 cột. Phần giả lập giữ nguyên nội dung, token, lượt thích,
ngày đăng, ngày cập nhật và các cột khác ngoài năm cột ID.
Số dòng, thứ tự dòng và schema Core 18 cột không thay đổi.

Một số bình luận cha vốn không nằm trong mẫu 73.247 dòng. Bản này giữ nguyên
tình trạng đó và ánh xạ cả các ID tham chiếu, không tự tạo thêm bình luận.

## File trong ZIP

```text
mixed_comments_73247_v3/
├── comments.parquet
├── comments.csv
├── videos/part-00000.parquet
├── id_mapping.csv
├── preview_100.csv
├── synthetic_preview_100.csv
├── summary.json
├── validation_report.json
├── remap_ids.py
└── README.md
```

`comments.parquet` và `comments.csv` là cùng một bảng trộn 73.247 dòng,
đúng 18 cột. CSV dùng UTF-8 BOM, cột tokens được ghi dạng JSON.
`videos/` chứa 500 video giả lập đã đổi ID để nối với phần giả lập trong bảng
bình luận. Video thật tiếp tục dùng metadata Core gốc.

`id_mapping.csv` là báo cáo đối chiếu ID cũ/mới nằm ngoài bảng bình luận,
vì vậy báo cáo này cố ý còn ID cũ có tiền tố `SYN_`. Không dùng tên hay dạng ID
để suy luận nguồn; dùng cột `source` vốn có.

Giải nén thư mục vào `data/mixed_comments_73247_v3/`. Khi lọc theo video, dùng
metadata giả lập đi kèm bản v3; metadata giả lập cũ vẫn mang ID cũ.
Kết quả lọc/GSDMM đã chạy trên v2 cũng vẫn mang ID cũ: hãy xử lý lại từ v3
hoặc nối qua `id_mapping.csv` nếu cần ghép kết quả theo ID.
Bản này chưa lọc lại nội dung hoặc chạy lại mô hình; chỉ thay ID.

## Kiểm tra

Đã đối chiếu toàn bộ 73.247 dòng, schema và metadata Parquet, xác nhận:

- 48.809 dòng thật không thay đổi.
- Đảo ánh xạ ID khôi phục chính xác toàn bộ bảng v2.
- Không trùng comment_id; liên kết thread/parent giữ nguyên.
- 500 video giả lập có metadata tương ứng với ID mới.
- CSV khớp Parquet; bảng bình luận không thêm cột.

`summary.json` ghi hash đầu vào trước/sau; `validation_report.json` ghi kiểm
tra độc lập. ID sinh bằng SHA256 với namespace riêng và kiểm tra va chạm với
ID của dữ liệu thật trong Core. Các nguồn, v2 và `curate.py` giữ nguyên.

## Tái tạo việc đổi ID

Chạy từ thư mục gốc repo, có sẵn v2, Core và metadata video giả lập gốc:

```powershell
docker run --rm -v "${PWD}:/project" -w /project gsdmm-local python3 datasets/mixed_comments_73247_v3/remap_ids.py --output data/mixed_comments_73247_v3_copy
```

Image cần có pyarrow. Script từ chối ghi đè thư mục đã tồn tại. Hash của ZIP
được ghi ở `SHA256.txt` cạnh file tải trên GitHub.
