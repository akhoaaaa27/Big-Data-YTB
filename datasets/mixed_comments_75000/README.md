# Bộ ghép 75.000 bình luận, đúng 18 cột Core

[Tải ZIP đầy đủ (khoảng 19 MB)](./ytb-mixed-comments-75000.zip?raw=true)

Bấm liên kết trên, hoặc mở file ZIP rồi chọn **Download raw file**.
Giải nén được thư mục `mixed_comments_75000/`.

[Xem trước 100 dòng](./preview_100.csv).

## Cách lấy mẫu và thời gian

- Chọn ngẫu nhiên **50.000 dòng thực tế từ Core** và **25.000 dòng từ bộ giả lập 50.000 dòng**.
- Lấy không lặp với seed 42; trộn thứ tự, tổng cộng 75.000 ID duy nhất.
- Giữ nguyên nội dung, ID, ngày đăng, ngày cập nhật và `published_date` của phần thực tế.
- Phần giả lập được phân ngày trong đúng khoảng thời gian của mẫu thật:
  **29/08/2026–21/10/2026**, UTC. Mỗi ngày có 462 hoặc 463 dòng giả lập.
- Tổng 75.000 dòng không phân bố đều theo ngày vì giữ nguyên ngày của phần thực tế.
- Giữ thứ tự cha trước reply khi cha có mặt trong phần giả lập được lấy mẫu.
- `processed_at` là thời điểm tạo bản ghép mới, không phải ngày đăng.

## Các file và schema

- `comments.parquet`: một file mới, đủ 75.000 dòng; cùng thứ tự, kiểu dữ liệu và 18 cột của Core.
- `comments.csv`: đủ 75.000 dòng và một dòng tiêu đề, UTF-8 BOM để mở bằng Excel; tokens là chuỗi JSON.
- `preview_100.csv`: 100 dòng xem nhanh.
- `summary.json`, `validation_report.json`, `README.md`: thông tin tạo mẫu và kiểm tra, tách khỏi bảng.

Không thêm cột nhãn, cờ hay cột phân biệt hai nguồn. Giá trị `source` trong cột
đã có giữ nguyên từ đầu vào (`api`, `legacy_csv`, `synthetic`). Metadata của bộ
trộn được ghi trong tài liệu này thay vì thêm vào bảng 18 cột.

## Lưu ý khi sử dụng

Bộ này có **25.000 bình luận giả lập theo mẫu**, không phải 75.000 bình luận thật.
Dữ liệu thật lấy từ Core gốc chưa lọc lại độ liên quan của video với nhạc AI.
Không suy ra tỷ lệ sentiment của cả bộ từ nhãn của phần giả lập.

Do lấy mẫu ngẫu nhiên từng dòng, có 4.445 reply không có bình luận cha trong mẫu;
ID cha, thread và tổng reply vẫn giữ từ nguồn. Hai nguồn gốc và `curate.py`
không bị sửa. Chỉ thay ngày đăng/cập nhật của phần giả lập trong bản trộn.

Đã kiểm tra 75.000 dòng CSV/Parquet, đúng schema 18 cột, ID duy nhất, bảo toàn
ngày và nội dung của toàn bộ 50.000 dòng thật, số token và quan hệ cha–con.
GSDMM chạy kiểm tra một vòng trên mẫu 1.000 dòng; đây không phải huấn luyện đầy đủ.

## Chạy GSDMM bằng Docker

Đặt thư mục đã giải nén vào `data/mixed_comments_75000/` trong repo, rồi chạy
trong Windows PowerShell ở thư mục repo (image `gsdmm-local` có numpy, pyarrow):

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/mixed_comments_75000/comments.parquet --output-dir /project/data/business/topics/mixed_seed_42 --seed 42
```

Mô hình bỏ dòng không modelable, reply, trùng nội dung và lọc từ hiếm nên số train
sẽ nhỏ hơn 75.000. Chọn output mới nếu đã chạy trước đó; không thay Core bằng bộ trộn.

SHA256 của ZIP:

```text
a2ea4225632b236cbe8e50877460e0271473286f67814f1839215fd6872e480e
```
