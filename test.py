import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("YOUTUBE_API_KEY")
print("Đã đọc key:", key[:6] + "..." if key else "KHÔNG ĐỌC ĐƯỢC KEY")