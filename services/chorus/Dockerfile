FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/chorus/ services/chorus/
EXPOSE 8009
CMD ["python", "-m", "services.chorus.main"]
