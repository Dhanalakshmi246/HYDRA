FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/evacuation_rl/ services/evacuation_rl/
EXPOSE 8011
CMD ["python", "-m", "services.evacuation_rl.main"]
