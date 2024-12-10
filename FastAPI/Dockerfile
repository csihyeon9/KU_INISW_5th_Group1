# Dockerfile
FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ssl-keyfile=/etc/letsencrypt/live/finwise.p-e.kr/privkey.pem", \
     "--ssl-certfile=/etc/letsencrypt/live/finwise.p-e.kr/fullchain.pem"]