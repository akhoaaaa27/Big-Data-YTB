# Dữ liệu trên máy chạy

Chép Parquet đã làm sạch vào `silver/comments_clean/`. Không chép dữ liệu gộp và các part tương đương cùng lúc. `.crc` và `_SUCCESS` không cần thiết cho các script này.

Pipeline không sửa dữ liệu Silver. Các đầu ra nằm trong `gold/`; Hugging Face cache nằm trong `hf-cache/`. Thư mục data có thể nặng, không đưa vào Docker build context hoặc commit Git.

Nếu dữ liệu gồm bình luận giả lập, ghi rõ điều đó trong mô tả bộ dữ liệu và báo cáo nghiên cứu. Không coi các thống kê từ tập đã trộn là bằng chứng đại diện cho toàn bộ người dùng thực tế.
