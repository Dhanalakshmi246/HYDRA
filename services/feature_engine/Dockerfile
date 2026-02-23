FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/feature_engine/ services/feature_engine/

EXPOSE 8003
CMD ["uvicorn", "services.feature_engine.main:app", "--host", "0.0.0.0", "--port", "8003"]
