FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/federated_server/ services/federated_server/
EXPOSE 8010
CMD ["python", "-m", "services.federated_server.main"]
