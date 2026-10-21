# Bộ ghép 73.247 dòng — đa dạng cấu trúc phần giả lập

[Tải ZIP đầy đủ](./ytb-mixed-comments-73247-v2.zip?raw=true)

[Xem trước 100 bình luận giả lập đã sửa](./synthetic_preview_100.csv).
[Xem trước bản ghép](./preview_100.csv).

Đây là **bộ hỗn hợp 48.809 dòng thật + 24.438 dòng giả lập theo mẫu**,
không phải toàn bộ bình luận YouTube thật. Giữ bản cũ ở folder riêng.
Không dùng bộ giả lập làm bằng chứng về cảm xúc người xem thật hoặc đánh giá
độ chính xác mô hình trên dữ liệu thật.

## Những gì đã thay đổi

Chỉ viết lại `text_raw`, làm sạch lại `text_clean`, tách lại `tokens`, tính lại
`n_tokens` và `is_modelable` cho phần giả lập. Giữ nguyên hoàn toàn 48.809 dòng
thật; giữ ID, tác giả, ngày đăng/cập nhật/processed_at, lượt thích và thread
của mọi dòng. Vẫn **đúng 18 cột Core**, không thêm cột nhãn/cờ/kiểu câu.

| Cấu trúc sinh | Số dòng giả lập |
|---|---:|
| Câu ngắn | 7.331 |
| Cảm thán | 3.666 |
| Câu hỏi | 4.888 |
| Nhận xét dài | 8.553 |

Ví dụ: “hay quá ad ơi”, “đỉnh thật!”, “có bản karaoke không ad?”, và nhận xét
nhiều câu về giọng hát, lời, phối khí, cảm xúc, bản quyền, quy trình làm nhạc.
Có câu hỏi ngắn nên nhóm kiểu câu là nhóm thiết kế khi sinh, không phải các
nhóm được người đánh giá phân loại độc lập. Vẫn sinh bằng mẫu/tổ hợp; các câu
ngắn có thể trùng tự nhiên, không thêm số hoặc mã vào câu để giả vờ khác nhau.

## Kiểm tra mở đầu lặp

Tính trên **24.438 dòng giả lập**, sau khi chuẩn hóa chữ hoa, dấu câu và dấu
tiếng Việt; không chỉ thay emoji để né kiểm tra.

| Cách đo | Tỷ lệ lớn nhất |
|---|---:|
| Một từ đầu | 4.00% |
| Hai từ đầu | 1.90% |
| Ba từ đầu | 0.76% |
| Năm từ đầu | 0.46% |
| Câu/mệnh đề mở đầu | 0.46% |

Các cụm mở đầu 2/3/5 từ và mệnh đề đầu đều dưới 2,5%, thấp hơn ngưỡng 3–5%.
Một từ đơn lẻ như “mình”, “phần”, “có” không phải cả câu mở đầu; vẫn được giữ
không quá 4%. Chi tiết tần suất và ví dụ nằm trong `validation_report.json`.

Phần giả lập có 18.651 `text_clean` khác nhau; câu đầy đủ lặp tối đa 24 lần
(khoảng 0,10%). Câu ngắn/cảm thán có thể thiếu token, được giữ cho sentiment và
đặt `is_modelable=False` khi có dưới 3 token, theo quy tắc repo. Vì vậy GSDMM
không mặc định dùng toàn bộ số dòng xuất.

Sentiment được thiết kế chủ yếu tích cực như bản trước: 17.140 tích cực,
4.872 trung lập, 2.426 tiêu cực trong phần giả lập. Đây là nhãn gán từ mẫu,
không phải đánh giá độc lập; không suy ra tỷ lệ cảm xúc của 48.809 dòng thật.
Một số phản ứng ngắn chung chung không biểu lộ khía cạnh cụ thể.

## File và sử dụng

ZIP có `comments.parquet`, `comments.csv`, hai file preview, hướng dẫn,
`summary.json` và `validation_report.json`. CSV dùng UTF-8 BOM, tokens dạng JSON.
Ngày UTC giữ nguyên 29/08/2026–21/10/2026. Nguồn thật là Core chưa lọc lại mức
liên quan video với nhạc AI; cột source có sẵn giữ đúng giá trị đầu vào.

Giải nén vào `data/mixed_comments_73247_v2/` trong repo rồi chạy PowerShell
với image gsdmm-local có numpy/pyarrow:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/mixed_comments_73247_v2/comments.parquet --output-dir /project/data/business/topics/mixed_73247_v2_seed_42 --seed 42
```

Đã kiểm tra toàn bộ dữ liệu, bảo toàn dòng thật và metadata, CSV/Parquet,
ngưỡng mở đầu; chạy GSDMM một vòng trên mẫu để kiểm tra luồng. Chưa huấn luyện
đầy đủ hay suy luận sentiment. `curate.py`, dữ liệu gốc và bản cũ giữ nguyên.

SHA256 của ZIP:

```text
df9ede1e49d048d9af5cdc8323e1f90e0f1f85f7c058a50e199a0dce89edbeab
```
