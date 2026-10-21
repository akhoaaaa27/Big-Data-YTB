# Lọc nguồn và đối chiếu GSDMM thật/giả

[Tải ZIP đầy đủ](./ytb-source-comparison-73247.zip?raw=true)

[Đọc kết quả và giới hạn](./RESULTS.md). [Xem bộ lọc](../../ytb/filter_comments.py).

| Kết quả | Thật | Giả lập |
|---|---:|---:|
| Đầu vào | 48.809 | 24.438 |
| Bình luận chuẩn bị cho GSDMM | 596 | 12.299 |
| GSDMM thực tế dùng | **594** | **12.299** |
| Bình luận chuẩn bị cho sentiment | **1.416** | 22.195 |

**Giữ hai nguồn riêng.** Giả lập chỉ dùng kiểm thử/đối chứng, không biến thành
ý kiến thật. Mô hình nhóm thật còn ít dữ liệu; không đủ cơ sở xác nhận 20 khía
cạnh AI. Các trường hợp chưa chắc được ghi trong audit, không xóa nguồn gốc.

Hai mô hình chính đã chạy đủ K20, alpha=beta=0,1, seed 42, 30 vòng; có thêm
K5 trên nhóm thật để xem ảnh hưởng của K khi mẫu nhỏ. Nhóm khác kích thước và
có cách tạo dữ liệu khác nhau nên không phải benchmark độ chính xác.

## Xem nhanh

- [Top words thật K20](./real_k20_top_words.csv)
- [Top words thật K5](./real_k5_top_words.csv)
- [Top words giả lập K20](./synthetic_k20_top_words.csv)
- [100 dòng thật sau lọc](./real_preview_100.csv)
- [100 dòng giả lập sau lọc](./synthetic_preview_100.csv)

ZIP giải nén thành `source_comparison/`. Có bảng bình luận 18 cột cho topics
và sentiment, video metadata và quyết định giữ/loại, audit ngôn ngữ, model,
top words và ví dụ ngẫu nhiên mỗi cụm. Không chạy suy luận sentiment ở bước này.
Dữ liệu gốc, bản 73.247 dòng và `curate.py` giữ nguyên.

## Dùng kết quả có sẵn

Đặt thư mục giải nén vào `data/source_comparison/`. Phân tích dữ liệu thật dùng
`real/topics/comments/part-00000.parquet`; sentiment thật dùng
`real/sentiment/comments/part-00000.parquet`. Không cần chạy lọc lại để xem kết quả.

Để train lại nhóm thật với K5 và output mới, dùng image gsdmm-local có numpy/pyarrow:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/source_comparison/real/topics/comments --output-dir /project/data/business/topics/real_filtered_k5_seed_42 --k 5 --seed 42 --no-vi-filter
```

`--no-vi-filter` ở đây vì đầu vào **đã lọc bằng Lingua**, tránh áp thêm cách
kiểm tra dấu cũ. Các bảng đầu vào vẫn đúng 18 cột; kết quả mô hình có thêm
cluster/text_gsdmm theo chức năng của mô hình.

## Tái chạy đầy đủ từ bản 73.247 dòng

Cần file `data/mixed_comments_73247_v2/comments.parquet`, Core videos và metadata
giả lập (đã có ở `data/source_comparison/synthetic/videos` sau giải nén). Chọn
thư mục output mới. Lệnh dưới cài bộ nhận diện offline vào container tạm:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local sh -c "pip install --no-cache-dir lingua-language-detector==2.1.1 && python3 -u -m ytb.filter_comments --compare-sources --train --synthetic-videos-dir /project/data/source_comparison/synthetic/videos --output-dir /project/data/source_comparison_2"
```

Lingua dùng điểm confidence >=0,8 và margin >=0,15; điểm này không phải độ
chính xác đã được kiểm bằng nhãn người đánh giá. Câu ngắn/emoji chưa rõ ngôn
ngữ chỉ giữ cho sentiment. Metadata không chứng minh được nơi sản xuất tại
Việt Nam hoặc âm thanh chắc chắn do AI tạo. Quyết định không chắc có báo cáo riêng.

Đã có 52 kiểm tra chức năng đạt, kiểm tra độc lập toàn bộ bảng bình luận và
chạy cả hai mô hình 30 vòng. Chưa đánh giá độ chính xác chủ đề hay sentiment.

SHA256 của ZIP:

```text
34e567d076e4d9c59ba662132c4311f731fabe9982738e6d6d1ce1641328ff0a
```
