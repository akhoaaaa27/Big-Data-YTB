# Thử chuẩn bị bình luận cho GSDMM và sentiment

## Đối chiếu phần thật và phần giả lập của bản 73.247 dòng

Chế độ mới của `ytb/filter_comments.py` đọc bản ghép, lọc riêng từng nguồn rồi
chạy hai mô hình GSDMM với cùng K=20, alpha=beta=0,1, seed 42, 30 vòng. Metadata
thật lấy từ Core; metadata giả lập chỉ là đầu vào đối chứng, không xác minh
âm thanh hoặc xuất xứ Việt Nam. Mô hình nhận số dòng khác nhau sau lọc, nên
đây là đối chiếu chẩn đoán, không phải phép đo độ chính xác hoặc so sánh nhân quả.

Cần có trong `data/`: `mixed_comments_73247_v2/comments.parquet`, `core/videos/`
và `synthetic_comments_50000/videos/`. Metadata giả lập nằm trong ZIP 50.000
dòng ở `datasets/synthetic_comments_50000/`; chỉ cần lấy thư mục `videos/`.

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local sh -c "pip install --no-cache-dir lingua-language-detector==2.1.1 && python3 -u -m ytb.filter_comments --compare-sources --train"
```

Kết quả mặc định ở `data/source_comparison/`:

```text
comparison.json
real/
  topics/comments/part-00000.parquet
  sentiment/comments/part-00000.parquet
  comments_audit.parquet
  video_decisions.csv
  summary.json
  gsdmm/seed_42/
synthetic/
  (cùng cấu trúc)
```

Các bảng **bình luận đầu vào topics/sentiment vẫn đúng 18 cột Core**; ID và lý
do lọc nằm trong audit riêng. Không thay văn bản, ID hay ngày của dòng được giữ.
Kết quả mô hình có thêm cluster/text_gsdmm theo chức năng GSDMM. Model tắt
heuristic dấu vì Lingua đã lọc ngôn ngữ ở bước trước. `topic_examples.json`
ghi tối đa 5 ví dụ ngẫu nhiên có seed mỗi cụm để đọc kèm top words.

Bỏ `--train` nếu chỉ muốn xuất dữ liệu và kiểm tra funnel. Khi output đã tồn
tại, chọn `--output-dir /project/data/source_comparison_2`. Không ghi đè Core,
bản ghép cũ hoặc kết quả trước. Câu ngắn/emoji vẫn được giữ cho sentiment;
chưa có suy luận sentiment trong bước chuẩn bị/đối chiếu này.

## Chạy lọc đơn lẻ

Có thể dùng file chạy riêng `ytb/filter_comments.py` để lọc rồi đưa ngay vào
GSDMM. Toàn bộ bước chuẩn bị bình luận nằm trong file này, dùng lại bộ phân loại
metadata có sẵn mà không sửa `ytb/curate.py`. Ví dụ
PowerShell (image `gsdmm-local` có numpy/pyarrow):

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local sh -c "pip install --no-cache-dir lingua-language-detector==2.1.1 && python3 -u -m ytb.filter_comments --train"
```

Lệnh xuất `data/filtered_comments/topics/comments/part-00000.parquet`, rồi
GSDMM đọc đúng thư mục đó. Kết quả mô hình nằm ở
`data/filtered_comments/gsdmm/seed_42/`. Bỏ `--train` nếu chỉ muốn lọc để xem
báo cáo trước. `--iterations 1` chỉ kiểm tra luồng, mặc định train 30 vòng.
Thư mục đầu ra đã tồn tại sẽ bị từ chối để bảo toàn kết quả; chọn
`--output-dir /project/data/filtered_comments_2` cho lần chạy mới.

File `ytb/filter_comments.py` đọc Core, phân loại video
bằng metadata nhạc AI, yêu cầu thêm dấu hiệu tiếng Việt, rồi chuẩn bị hai tập
bình luận riêng. Không thu thập lại, không sửa Core,
`curated`, `review` hoặc kết quả mô hình cũ. Chạy lại phải chọn thư mục xuất mới.
Riêng bản thử yêu cầu bằng chứng trong mô tả phải có dạng credit hoặc công bố
gắn với tác phẩm; không nhận danh sách quảng bá kiểu “AI Cover độc lạ”.

Đây là thử nghiệm dựa trên metadata, chưa xác minh âm thanh do AI tạo hoặc nơi
sản xuất tại Việt Nam. Metadata khai báo ngôn ngữ âm thanh khác tiếng Việt sẽ
chuyển video sang `uncertain`, dù tiêu đề/mô tả được dịch sang tiếng Việt.
Nếu không khai báo ngôn ngữ âm thanh, bộ nhận diện kiểm tra tiêu đề hoặc tối đa
1.000 ký tự đầu mô tả. Những thông tin này có thể sai hoặc nói về phần khác
của video. Không dùng vị trí tìm kiếm, tên kênh để chứng minh xuất xứ Việt Nam.

## Chạy thử trên PowerShell

Mở terminal ở thư mục repo, Docker Desktop đang chạy. Image `gsdmm-local` cần
có `numpy`, `pyarrow` như lần train trước. Lệnh dưới đây cài thêm Lingua vào
container tạm rồi chạy; thư viện có sẵn mô hình, không gửi bình luận ra API.
Lần chạy tiếp theo container mới sẽ cài lại thư viện, có thể thêm dòng
`RUN pip install --no-cache-dir lingua-language-detector==2.1.1` vào Dockerfile
và build image khi muốn dùng lâu dài.

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local sh -c "pip install --no-cache-dir lingua-language-detector==2.1.1 && python3 -u -m ytb.filter_comments"
```

Chỉ sao chép lệnh, không chép `PS ...>` hoặc `>>`. Bộ nhận diện tất cả ngôn ngữ
cần khoảng vài GB RAM; chạy có thể mất vài phút. Đầu ra:

```text
data/filtered_comments/
  summary.json
  video_decisions.csv
  comments_audit.parquet
  videos/part-00000.parquet
  video_snapshots/part-00000.parquet  (nếu nguồn có)
  topics/comments/part-00000.parquet
  sentiment/comments/part-00000.parquet
```

Các dòng bị loại vẫn còn trong Core và `comments_audit.parquet`. Hai cột
`topic_reason` và `sentiment_reason` ghi lý do độc lập; `keep` là được giữ.
Báo cáo có số dòng từng nhóm, độ tập trung theo video và SHA256 nguồn trước/sau.
Đường dẫn mặc định lấy từ `YTB_DATA_DIR`, có thể thay bằng
`python -m ytb.filter_comments --source-dir /project/data/core --output-dir /project/data/filtered_comments_2`.

## Quy tắc của bản thử

- Ngôn ngữ dùng Lingua 2.1.1 cho tất cả ngôn ngữ nói, yêu cầu confidence ≥ 0,8
  và chênh lệch hai ngôn ngữ đứng đầu ≥ 0,15. Đây là điểm của bộ nhận diện,
  không phải độ chính xác đã đo trên dữ liệu này.
- Câu dưới 3 từ hoặc dưới 12 chữ cái được đánh dấu `short`, chưa xác định
  ngôn ngữ. Câu tiếng Việt không dấu có thể rơi vào `uncertain`, không tự nhận
  là tiếng Việt. Không dùng cách chỉ thấy ký tự có dấu là coi tiếng Việt.
- Sentiment giữ bình luận tiếng Việt nhận diện đủ ngưỡng và nhóm ngắn/emoji
  chưa rõ ngôn ngữ; giữ replies, trùng văn bản và không giới hạn theo video.
  Nhóm ngắn vì vậy vẫn có thể chứa tiếng nước ngoài. Không xóa lời chào,
  tên ca sĩ, từ tục hoặc từ phủ định chỉ vì chúng xuất hiện.
- Chỉ nghi spam khi có URL kèm cụm mời mua hàng/kiếm tiền/cá cược rõ ràng.
  Quy tắc này còn có thể nhầm và bỏ sót; các dòng được cách ly để kiểm tra lại.
- GSDMM yêu cầu tiếng Việt đủ ngưỡng, `is_modelable`, không phải reply,
  ít nhất hai token, bỏ trùng `text_clean`, tối đa 50 bình luận/video.
  Chọn bằng hash cố định ID, không tùy thứ tự file Parquet. Đổi giới hạn bằng
  `--cap-per-video`. Giới hạn giảm ảnh hưởng video lớn nhưng cũng đổi phân bố
  mẫu; không dùng mẫu đã giới hạn để ước lượng tỷ lệ sentiment toàn bộ.

## Train sau khi xem báo cáo

```powershell
docker run --rm -v "${PWD}:/project" -w /project -e YTB_DATA_DIR=/project/data gsdmm-local python3 -u -m ytb.modeling.train_topics --input /project/data/filtered_comments/topics/comments --output-dir /project/data/business/topics/trial_seed_42 --seed 42
```

`train_topics.py` đọc cột `comment_language_status` của đầu vào đã chuẩn bị,
không lọc dấu tiếng Việt thêm lần nữa. Nếu đọc Core cũ, hành vi mặc định trước
đây vẫn còn. Mô hình có thể bỏ thêm dòng sau khi lọc từ hiếm và token, xem
`training_summary.json`; số đầu vào thử chưa phải số dòng cuối cùng huấn luyện.

Sentiment dùng `--input /project/data/filtered_comments/sentiment/comments` và
`--text-column text_raw` để giữ câu gốc, emoji và phủ định. Phải chọn model
sentiment tiếng Việt và kiểm tra model card/ánh xạ nhãn trước khi chạy.
Chưa chạy suy luận sentiment trong thử nghiệm chuẩn bị dữ liệu này.
