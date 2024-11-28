FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ssl-keyfile=/etc/letsencrypt/live/finwise.p-e.kr/privkey.pem", \
     "--ssl-certfile=/etc/letsencrypt/live/finwise.p-e.kr/fullchain.pem"]