# GSDMM và sentiment cho bình luận nhạc AI tiếng Việt

Quy trình tiếp nối dữ liệu `comments_clean` bạn đã làm:

```text
data/silver/comments_clean/*.parquet
    → GSDMM: mỗi bình luận một cluster
    → Đặt tên cluster: mỗi bình luận một aspect
    → VisoBERT: thêm sentiment cho bình luận
    → data/gold/sentiment_seed42/comments_sentiment.parquet + CSV
```

**Mỗi bình luận chỉ có một dòng và một khía cạnh chính trong cột `aspect`.** Khía cạnh được kế thừa từ cụm GSDMM, không chuyển sang cách gán nhiều khía cạnh bằng từ khóa. Cụm chưa rõ hoặc không liên quan được gán `Khác`. Sentiment vẫn được dự đoán cho các dòng `Khác`, không ép thành trung lập.

Repo bắt đầu từ Silver đã làm sạch; không thu thập lại YouTube và không sửa `curate.py` hay Parquet gốc. Repo public chứa code và bảng tên cụm; bạn tự chép dữ liệu vào máy chạy.

Repo đã có thư mục `test/` với code và dữ liệu bạn tải lên trước đó; phần đó được giữ nguyên. Bộ chạy này nằm riêng trong **`gsdmm_sentiment/`**, cạnh `test/`. Chạy mọi lệnh tại `GSDMMnSentiment\gsdmm_sentiment`, nơi có `jobs/`, `config/`, `data/` và `compose.yaml` cùng cấp với README này. Dữ liệu và code trong `test/` không được đưa vào Docker build context của bộ chạy mới.

## 1. Chuẩn bị trên Windows

Cài Git và Docker Desktop, mở Docker Desktop và dùng Linux containers. Mở **PowerShell**, chạy từng lệnh dưới đây. Không chép phần dấu nhắc `PS ...>` hoặc `>>` vào lệnh.

```powershell
git clone https://github.com/tnhminh237/GSDMMnSentiment.git
cd .\GSDMMnSentiment\gsdmm_sentiment
docker version
docker compose version
```

Nếu tải ZIP từ GitHub, giải nén rồi mở PowerShell tại thư mục chứa `README.md` và `compose.yaml`.

## 2. Chép dữ liệu đã clean vào đúng chỗ

Chép các file **`.parquet`** của thư mục cũ vào `data\silver\comments_clean`. Ví dụ đúng theo đường dẫn dự án bạn đã dùng:

```powershell
Copy-Item -Path "D:\DUE\NAM 2\NMDLLL\PROJECT\test\data\silver\comments_clean\*.parquet" -Destination .\data\silver\comments_clean\
Get-ChildItem .\data\silver\comments_clean\*.parquet
```

Chép dữ liệu một lần vào thư mục trống. Đừng chép cả một file gộp và các file part chứa cùng dữ liệu. Script đọc toàn bộ `*.parquet` trong thư mục; một file hay bốn file đều được. Không cần chép `.crc` hoặc `_SUCCESS`.

Silver của bạn có các cột:

| Cột | Mục đích |
|---|---|
| `comment_id` | ID bình luận, khóa để giữ mỗi bình luận một dòng |
| `video_id` | Video chứa bình luận |
| `author` | Tác giả, giữ lại nếu có |
| `text` | Văn bản cho sentiment và xem mẫu |
| `likes` | Lượt thích, giữ lại nếu có |
| `published` | Ngày đăng, giữ lại nếu có |
| `tokens` | Danh sách token đã tách từ cho GSDMM, ví dụ `["giai_điệu", "hay"]` |

Bốn cột bắt buộc là `comment_id`, `video_id`, `text`, `tokens`. Các cột còn lại được giữ nếu có. Dùng Parquet Silver để train, không dùng CSV `peek_comments_clean.csv`.

## 3. Build Docker image

Đứng tại thư mục `GSDMMnSentiment\gsdmm_sentiment`:

```powershell
docker compose build gsdmm sentiment
```

Lần đầu cần Internet để tải thư viện. Hai image được tạo là `gsdmm-local` và `sentiment-local`. File Dockerfile là cấu hình được Docker đọc; không chạy các dòng `FROM`, `RUN` trong PowerShell.

## 4. Chạy GSDMM

```powershell
docker compose run --rm gsdmm
```

Mặc định: seed `42`, tối đa `20` cụm, `30` vòng, alpha/beta `0.1`, tần suất từ tối thiểu `2`. Chương trình in từng vòng để theo dõi.

Kết quả:

```text
data/gold/
├── comments_clustered.parquet
├── cluster_top_words.csv
├── gsdmm_model.npz
└── training_summary.json
```

`comments_clustered.parquet` có cột `cluster`. **Mã cụm chưa phải tên khía cạnh** và không phải nhãn sentiment.

Các bước giảm số dòng trong GSDMM được ghi trong `training_summary.json`:

- Bỏ dòng thiếu ID/video/text hoặc text trống; bỏ ID trùng để có khóa duy nhất.
- Bỏ reply nếu đầu vào còn cột `is_reply`; Silver hiện tại đã bỏ reply ở bước trước.
- Bỏ văn bản trùng sau chuẩn hóa để câu lặp không chi phối cụm.
- Bỏ từ có tổng tần suất dưới `2`; bỏ bình luận còn dưới `2` token sau bước từ điển vì quá ít thông tin để phân cụm.

**Không có bước yêu cầu `n_tokens >= 3`, không lọc lại theo dấu tiếng Việt, không giới hạn bình luận mỗi video.** Script không ghi ngược vào Silver.

Lần chạy trước của bạn có `40.701` dòng Silver và `34.923` dòng sau GSDMM. Đây là số tham khảo của dữ liệu cũ, không phải số bị ép trong code; đổi dữ liệu hoặc thư viện có thể cho cụm khác.

## 5. Gán đúng một khía cạnh cho mỗi bình luận

### Dùng bảng tên cụm của lần chạy đã xem

```powershell
docker compose run --rm aspects
```

Script dùng `config/cluster_aspects_seed42.json`, chứa bảng tên cụm của lần chạy `34.923` bình luận bạn đã xem. Bảng từ khóa tham khảo nằm tại `config/cluster_top_words_seed42.csv`.

Script kiểm tra dấu kiểm tra SHA-256 của bảng từ khóa và số dòng mỗi cụm. **Nếu trùng lần chạy tham khảo**, kết quả nằm tại:

```text
data/gold/aspects_seed42/
├── comments_with_aspects.parquet
├── comments_with_aspects.csv
├── cluster_labels.csv
├── aspect_summary.csv
└── label_summary.json
```

Mỗi `comment_id` chỉ có một dòng, thêm một chuỗi `aspect` duy nhất. Nhiều cụm có thể mang cùng một tên khía cạnh. Nếu bình luận thực tế nói nhiều ý, quy trình này vẫn giữ khía cạnh chính theo cụm.

### Nếu báo “Bảng từ khóa khác lần train đã lập nhãn”

Đây là chặn dùng nhầm tên cụm cũ. Không xóa kiểm tra hash và không tự giữ nghĩa của mã cụm từ lần train khác. Tạo bảng tên cho **lần train mới**:

```powershell
docker compose run --rm aspects --prepare-mapping /opt/data/gold/cluster_aspects_review.json
notepad .\data\gold\cluster_aspects_review.json
```

Trong JSON, sửa giá trị của từng mã cụm ở `cluster_aspects` thành **một tên khía cạnh**. Ví dụ `"7": "Chất lượng âm thanh"`. Dựa vào `top_words` và các bình luận mẫu GSDMM in trong terminal; khi cần, đọc thêm bình luận của cụm trong `comments_clustered.parquet`. Cụm không rõ thì giữ `Khác`. Phần `review_context` có từ khóa và số bình luận để đối chiếu; đừng sửa `cluster_words_signature` để vượt kiểm tra.

Bạn đang đặt tên cho tối đa 20 cụm, không phải dán nhãn tay từng bình luận. Sau khi lưu:

```powershell
docker compose run --rm aspects --mapping /opt/data/gold/cluster_aspects_review.json
```

Các nhãn khía cạnh là cách diễn giải cụm, chưa được xác minh riêng từng bình luận. Cụm có nội dung lẫn nhau vẫn có thể gán sai khía cạnh cho một số dòng.

## 6. Chạy thử sentiment với 100 dòng

```powershell
docker compose run --rm sentiment --limit 100 --output-dir /opt/data/gold/sentiment_test100
```

Mở `data\gold\sentiment_test100\comments_sentiment.csv` để xem `text`, `aspect`, `sentiment`. Đây là 100 dòng đầu để kiểm tra quy trình, không phải mẫu ngẫu nhiên đại diện cho toàn bộ dữ liệu.

Mô hình là `5CD-AI/Vietnamese-Sentiment-visobert`. Lần đầu cần Internet tải model từ Hugging Face; cache nằm tại `data/hf-cache`, dùng lại cho những lần chạy sau. Máy chạy CPU, không cần GPU hoặc cài Python trên Windows.

## 7. Chạy sentiment toàn bộ

```powershell
docker compose run --rm sentiment
```

Chờ tiến độ `Sentiment: .../...` và dòng `Đã lưu sentiment tại: ...`. Kết quả:

```text
data/gold/sentiment_seed42/
├── comments_sentiment.parquet
├── comments_sentiment.csv
├── aspect_sentiment_summary.csv
└── sentiment_summary.json
```

**Dùng `comments_sentiment.parquet` hoặc CSV cùng tên để phân tích khía cạnh và sentiment.** Một dòng là một bình luận; có `cluster`, một `aspect` và một `sentiment`. Không tạo bảng một bình luận nhiều dòng theo khía cạnh.

| Cột thêm | Ý nghĩa |
|---|---|
| `sentiment` | `Tích cực`, `Trung lập` hoặc `Tiêu cực` |
| `sentiment_score` | Điểm softmax của nhãn được chọn; không phải độ chính xác đã kiểm chứng |
| `sentiment_negative` | Xác suất mô hình cho tiêu cực |
| `sentiment_neutral` | Xác suất mô hình cho trung lập |
| `sentiment_positive` | Xác suất mô hình cho tích cực |
| `sentiment_truncated` | `True` nếu văn bản bị cắt khi đưa vào model |

Sentiment dùng `text`, giữ từ phủ định và emoji, không dùng tokens đã bỏ stopword. Nhãn được đọc từ cấu hình model; không đoán thứ tự `0/1/2`. Bình luận không còn chữ sau chuẩn hóa vẫn được giữ với sentiment trống, không ép trung lập.

Mặc định model đọc tối đa **128 token**, gồm token đặc biệt. Không xóa hay cắt `text` gốc trong file xuất. Câu dài có thể mất ngữ cảnh cuối; số câu này được ghi trong `sentiment_summary.json`. Nếu muốn thử tối đa 256 token, dùng thư mục mới:

```powershell
docker compose run --rm sentiment --max-length 256 --output-dir /opt/data/gold/sentiment_seed42_len256
```

Đây là **sentiment chung của bình luận**, gắn cùng khía cạnh chính của cụm. Không được hiểu là mô hình đã xác định chính xác cảm xúc riêng hướng đến khía cạnh đó hoặc thái độ đối với AI. Repo không huấn luyện lại VisoBERT và chưa có đánh giá accuracy độc lập trên bộ bình luận này.

## Đã có kết quả từ dự án cũ thì có cần chạy lại không?

Không cần. Để tiếp tục đúng lần chạy cũ, chép các file phù hợp vào repo mới:

- Có `comments_clustered.parquet` và `cluster_top_words.csv`: chép cả hai cùng lần train vào `data/gold`, bắt đầu bước 5.
- Có `aspects_seed42/comments_with_aspects.parquet`: chép nguyên thư mục `aspects_seed42` vào `data/gold`, bắt đầu bước 6 hoặc 7.
- Đã có kết quả sentiment đầy đủ: dùng luôn để phân tích, không cần chạy lại.

Không trộn bảng từ khóa của lần train này với bình luận đã phân cụm của lần train khác.

## Chạy lại mà giữ kết quả cũ

Các script từ chối ghi đè kết quả. Đổi thư mục xuất và nối đúng đường dẫn giữa các bước. Ví dụ một lần chạy mới vẫn seed 42:

```powershell
docker compose run --rm gsdmm 42 --output-dir /opt/data/gold/run2
docker compose run --rm aspects --top-words /opt/data/gold/run2/cluster_top_words.csv --prepare-mapping /opt/data/gold/run2/cluster_aspects_review.json
notepad .\data\gold\run2\cluster_aspects_review.json
```

Đặt tên cụm trong JSON rồi chạy:

```powershell
docker compose run --rm aspects --input /opt/data/gold/run2/comments_clustered.parquet --top-words /opt/data/gold/run2/cluster_top_words.csv --mapping /opt/data/gold/run2/cluster_aspects_review.json --output-dir /opt/data/gold/run2/aspects
docker compose run --rm sentiment --input /opt/data/gold/run2/aspects/comments_with_aspects.parquet --output-dir /opt/data/gold/run2/sentiment
```

Nếu đổi seed khác 42, tên file GSDMM có hậu tố `_seedN`; truyền đúng tên đó vào `--input` và `--top-words`.

## Lệnh Docker thường nếu không dùng Compose

Chỉ chọn một cách chạy, không cần chạy lặp lại cả hai cách. Chạy các lệnh tại **`gsdmm_sentiment/`**, dùng chính thư mục này làm build context (dấu chấm cuối lệnh):

```powershell
docker build -t gsdmm-local -f .\jobs\Dockerfile.gsdmm .
docker build -t sentiment-local -f .\jobs\Dockerfile.sentiment .
docker run --rm --user 0 -v "${PWD}/jobs:/opt/jobs:ro" -v "${PWD}/config:/opt/config:ro" -v "${PWD}/data:/opt/data" gsdmm-local python3 -u /opt/jobs/gsdmm_train.py 42
docker run --rm --user 0 -v "${PWD}/jobs:/opt/jobs:ro" -v "${PWD}/config:/opt/config:ro" -v "${PWD}/data:/opt/data" gsdmm-local python3 -u /opt/jobs/assign_aspects.py
docker run --rm --user 0 -v "${PWD}/jobs:/opt/jobs:ro" -v "${PWD}/data:/opt/data" sentiment-local python3 -u /opt/jobs/predict_sentiment.py
```

## Cấu trúc thư mục `gsdmm_sentiment/`

```text
.
├── README.md
├── compose.yaml
├── jobs/
│   ├── gsdmm_train.py
│   ├── assign_aspects.py
│   ├── predict_sentiment.py
│   ├── Dockerfile.gsdmm
│   └── Dockerfile.sentiment
├── config/
│   ├── cluster_aspects_seed42.json
│   └── cluster_top_words_seed42.csv
├── requirements/
│   ├── gsdmm.txt
│   ├── sentiment.txt
│   └── dev.txt
├── tests/
└── data/
    ├── README.md
    ├── silver/comments_clean/       # Chép Parquet vào đây
    ├── gold/                        # Kết quả từng bước
    └── hf-cache/                    # Model tự tải khi chạy sentiment
```

## Kiểm tra code

GitHub Actions chạy các kiểm tra GSDMM → một aspect → ghi sentiment, nhãn model và việc giữ nguyên dữ liệu nguồn. Các kiểm tra sentiment dùng mô hình giả định để xác minh logic, **không đo chất lượng VisoBERT** và không tải model trong CI.

Chạy trên môi trường Python phát triển:

```powershell
python -m pip install -r .\requirements\dev.txt
python -m pytest -q
docker compose config --quiet
```

Không cần làm phần này nếu chỉ chạy pipeline qua Docker.

Kết quả kiểm tra và giới hạn của lần đóng gói được ghi tại [VALIDATION.md](VALIDATION.md).

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `gsdmm-local` hoặc `sentiment-local` không tồn tại | Chạy bước build trước |
| Không kết nối được Docker daemon | Mở Docker Desktop, kiểm tra Linux containers |
| Không thấy Parquet | Kiểm tra file đã nằm trực tiếp trong `data/silver/comments_clean` |
| `flit_core`/`inconsistent Name` khi cài Torch | Dùng Dockerfile trong repo: đã nâng pip và thêm PyPI cho dependency |
| Không tải model Hugging Face được | Kiểm tra Internet/proxy và quyền truy cập Hugging Face; chạy lại sau khi kết nối được |
| `Đã có kết quả` / `Đã có ...` | Kết quả cũ được giữ; chọn `--output-dir` khác |
| Cảnh báo token quá dài | Bản này đếm độ dài không phát cảnh báo thừa; đầu vào model vẫn cắt theo `--max-length`, có cờ `sentiment_truncated` |
| VS Code báo thiếu import | Máy Windows không có thư viện Python; script vẫn chạy trong Docker nếu image đã build |

## Dữ liệu và diễn giải kết quả

Thư mục `data/` của bộ chạy mới được bỏ qua bởi `.gitignore`, nên bình luận, model và cache tạo ở đây không được commit. Các file bạn đã tải lên trong `test/` được giữ nguyên; `.gitignore` không loại các file đã được Git theo dõi. Nếu dùng tập đã trộn dữ liệu thật và giả lập từ công việc trước, phải mô tả việc trộn đó trong báo cáo; tỷ lệ sentiment không đại diện tự động cho người dùng YouTube thực tế. Bộ chạy này không thêm bước tách nguồn hoặc tạo thêm dữ liệu giả.

Giữ nguyên ba tên file bạn đang chạy. Thay đổi khi đóng gói: bảng tên cụm được đưa ra JSON có kiểm tra cùng lần train; GSDMM chặn ghi đè; Dockerfile sửa lỗi dependency; đếm độ dài sentiment không phát cảnh báo gây nhầm lẫn, vẫn giữ cờ cắt văn bản.
