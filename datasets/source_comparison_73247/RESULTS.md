# Kết quả lọc và đối chiếu nguồn

Dùng đúng bản ghép 73.247 dòng v2, không thay dữ liệu gốc và không trộn phần
giả lập vào mô hình dùng để diễn giải phản ứng thật.

| Bước | Dòng thật | Dòng giả lập |
|---|---:|---:|
| Đầu vào | 48.809 | 24.438 |
| Thuộc video được nhận theo metadata | 1.814 | 24.438 (metadata mô phỏng) |
| Đầu vào GSDMM sau lọc | 596 | 12.299 |
| Thực tế train sau lọc từ hiếm | 594 | 12.299 |
| Đầu vào sentiment | 1.416 | 22.195 |

## Điều đã kiểm tra

- Video thật: 212 video có metadata đủ điều kiện; 1.119 video chưa đủ bằng
  chứng, 20 video bị loại bởi quy tắc nội dung. Không khẳng định toàn bộ
  nhóm chưa chắc đều không phải nhạc AI. Có báo cáo để xem và khôi phục.
- Lingua 2.1.1 chạy offline; confidence >=0,8 và margin >=0,15. Đây là điểm
  bộ nhận diện, không phải tỷ lệ chính xác được đo bằng nhãn người đánh giá.
- Sau khi video được nhận, 389 bình luận thật chưa chắc ngôn ngữ và 7 dòng
  được nhận diện chắc là ngôn ngữ khác được để ngoài mẫu phân tích.
- Câu ngắn/emoji chưa rõ ngôn ngữ giữ cho sentiment; không dùng cho GSDMM.
- GSDMM bỏ reply, trùng nội dung, dòng không modelable; cap 50 dòng/video.
  Không xóa tên ca sĩ, từ tục hoặc lời chào chỉ để làm đẹp cụm.
- Hai mô hình chính dùng cùng K=20, alpha=beta=0,1, seed 42, 30 vòng. Nguồn
  và kích thước mẫu vẫn khác nhau, nên đây không phải phép so sánh nhân quả
  hay benchmark độ chính xác. Chưa có nhãn người đánh giá cho chủ đề.

## Đọc kết quả

Dữ liệu thật không còn tạo các cụm lớn về gameshow, phim tổng tài hay toán
như bản ghép trước trong bảng top words được xem. Tuy nhiên nhiều cụm vẫn
là phản ứng chung, tên tác phẩm, kỷ niệm, trích ca từ, trò chuyện và mời
đăng ký kênh. Chưa được coi mọi bình luận trên video nhạc AI là ý kiến về
công nghệ AI. Vẫn có thể có câu lẫn ngôn ngữ hoặc metadata sai.

K=20 trên 594 dòng tạo nhiều cụm nhỏ: 8 cụm dưới 10 dòng, 2 cụm chỉ 2 dòng.
Thử thêm K=5 cho nhóm thật (cùng seed, 30 vòng) cho các nhóm rộng như
khen/ủng hộ/karaoke, tên tác phẩm/kỷ niệm, trích ca từ và trò chuyện. Đây
là cách đọc sơ bộ từ top words và ví dụ, không phải 5 khía cạnh đã xác nhận.

Nhóm giả lập cho các cụm dễ đọc hơn về phối khí, bè, phát âm, nhịp, hình
ảnh và bản quyền. Điều này phản ánh các chủ đề được đưa vào mẫu sinh;
không dùng làm bằng chứng đã phát hiện các khía cạnh đó trong khán giả thật.
Vẫn có cụm về bối cảnh nghe và câu mẫu. Không gộp lại để tuyên bố dữ liệu
thật có cùng cấu trúc, cũng không bù số lượng bằng dữ liệu giả lập.

Sau lọc, nhóm thật còn 146 video ở bước train. Video lớn nhất chiếm 4,04%,
10 video lớn nhất chiếm 28,62%. Không khẳng định mọi chỉ số đều cải thiện:
độ tập trung top 10 cao hơn vì tập thật sau lọc hẹp hơn.

## Sử dụng file nào

- Phân tích chủ đề trên phản ứng thật: `real/topics/comments/part-00000.parquet`
  (596 dòng đầu vào; mô hình dùng 594 dòng).
- Chuẩn bị sentiment thật: `real/sentiment/comments/part-00000.parquet`
  (1.416 dòng). Đây là đầu vào, chưa chạy suy luận sentiment.
- `synthetic/` chỉ phục vụ đối chứng và kiểm thử.
- Các bảng bình luận đầu vào đều đúng 18 cột; audit và báo cáo nằm riêng.
- Kết quả K20: `<nguồn>/gsdmm/seed_42/cluster_top_words.csv`,
  `comments_clustered.parquet`, `topic_examples.json`, `training_summary.json`.
- Thử K5 thật: `real/diagnostics/k5_seed_42/`.

Bước tăng quy mô dữ liệu thật cần thu thập thêm từ các video được xác nhận
phù hợp. Việc hạ ngưỡng lọc hoặc thay bằng dữ liệu sinh không chứng minh
thêm được phản ứng thật. Không xác minh âm thanh thực tế hay nơi sản xuất
Việt Nam chỉ bằng metadata tiếng Việt.
