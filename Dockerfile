# Image Airflow dùng chung cho cả nhóm, ghim đúng phiên bản 3.3.2.
FROM apache/airflow:3.3.2

# Spark chạy trên JVM nên cần Java. default-jre-headless lấy bản Java mặc định của hệ điều hành
# trong image (PySpark 4.x chạy được với Java 17 và 21).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt
# Ghim "apache-airflow==3.3.2" để pip không nâng/hạ phiên bản Airflow khi cài thêm thư viện.
RUN pip install --no-cache-dir "apache-airflow==3.3.2" -r /requirements.txt