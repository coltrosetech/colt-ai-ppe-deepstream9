FROM python:3.12-slim

WORKDIR /app

COPY video_selector/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY video_selector /app/video_selector

EXPOSE 8080

CMD ["uvicorn", "video_selector.app:app", "--host", "0.0.0.0", "--port", "8080"]

