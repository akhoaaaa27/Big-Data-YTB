# Bình luận đã lọc, gộp chung để chạy GSDMM và sentiment

[Tải ZIP đầy đủ](./ytb-combined-filtered-comments.zip?raw=true).
[Xem top words của mô hình chung](./cluster_top_words.csv).
[Xem 100 dòng đầu vào](./preview_100.csv).

Dữ liệu thật và giả lập đã được trộn trong cùng bảng và dùng một mô hình
GSDMM chung. Không có thư mục hay mô hình riêng theo nguồn trong ZIP này.
Mỗi bảng đầu vào giữ đúng 18 cột Core, mọi nội dung, ID và ngày đăng giữ nguyên.
Không thêm cột đánh dấu; cột `source` vốn có giữ giá trị ban đầu.

| Dữ liệu | Số dòng | Dùng cho |
|---|---:|---|
| `topics/comments/part-00000.parquet` | 12.895 | Đầu vào GSDMM chung |
| `sentiment/comments/part-00000.parquet` | 23.611 | Chuẩn bị cho dự đoán cảm xúc |

Hai bảng phân theo mục đích xử lý. Topics đã chọn tiếng Việt đủ ngưỡng,
`is_modelable=True`, không reply, khử trùng nội dung và giới hạn 50 dòng/video
ở bước lọc trước. Sentiment còn giữ câu ngắn, emoji và reply nên nhiều dòng hơn;
không khử trùng câu của những người dùng khác nhau. Có bản CSV tương ứng cho cả hai.
**Chưa chạy dự đoán sentiment**; bảng sentiment không chứa nhãn hoặc điểm cảm xúc.

ZIP giải nén thành `combined_filtered_comments/`. Đặt thư mục đó trong `data/`.
Mô hình đã chạy có trong `gsdmm/seed_42/`, gồm bình luận gán cluster, top words,
trọng số mô hình, báo cáo huấn luyện và ví dụ ngẫu nhiên từng cụm. Chỉ bảng kết quả
mô hình thêm hai cột `cluster` và `text_gsdmm`; bảng đầu vào vẫn đúng 18 cột.

## Train lại bằng Docker trên PowerShell

Chạy tại thư mục gốc repository. Cần image `gsdmm-local` đã có numpy và pyarrow.
Lệnh một dòng dưới tạo kết quả mới, giữ nguyên mô hình đã tải:

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/combined_filtered_comments/topics/comments --output-dir /project/data/business/topics/combined_filtered_seed42 --seed 42 --k 20 --iterations 30 --no-vi-filter
```

`--no-vi-filter` vì đầu vào đã qua bộ lọc ngôn ngữ Lingua ở bước trước.
Đừng bỏ tùy chọn này khi dùng tập đã lọc: heuristic dấu cũ có thể loại thêm câu hợp lệ.
Chọn output khác nếu thư mục kết quả đã tồn tại.

## Tái tạo việc ghép

Nếu đã có `data/source_comparison/` từ lần lọc trước, chạy lệnh sau để ghép
hai nhóm vào một output mới, không phải chạy lại nhận diện ngôn ngữ:

```powershell
docker run --rm -v "${PWD}:/project" -w /project gsdmm-local python3 datasets/combined_filtered_comments/combine_filtered.py --input data/source_comparison --output data/combined_filtered_comments_2
```

Script kiểm tra schema, ID duy nhất, trộn thứ tự bằng seed 42, từ chối ghi đè
output cũ và ghi hash đầu vào trước/sau. Không sửa `curate.py` hay dữ liệu Core.
`validation_report.json` kiểm tra độc lập rằng toàn bộ 18 cột và giá trị sau ghép
khớp đầu vào; `merge_summary.json` ghi số lượng, nguồn và tham số ở báo cáo riêng.

## Phạm vi sử dụng

Trong 12.895 dòng topics có 596 dòng thật và 12.299 dòng giả lập (95,4%).
Trong 23.611 dòng sentiment có 1.416 dòng thật và 22.195 dòng giả lập.
Việc gộp đáp ứng mục đích chạy thử chung; tỷ lệ chủ đề và cảm xúc trên tập này
chịu ảnh hưởng lớn của dữ liệu giả lập, không đại diện riêng cho người xem thật.
Metadata video chưa xác minh âm thanh AI hoặc nơi sản xuất tại Việt Nam.
Chưa đánh giá độ chính xác chủ đề và sentiment bằng nhãn độc lập.

Kết quả chạy chung: **12,893 bình luận huấn luyện**, 20 cụm hoạt động, đủ 30 vòng. Funnel và số dòng theo nguồn sau huấn luyện có trong `validation_report.json`.
