FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 업로드/결과 디렉토리
RUN mkdir -p uploads results

EXPOSE 8000

ENV PORT=8000

CMD uvicorn server:app --host 0.0.0.0 --port $PORT
