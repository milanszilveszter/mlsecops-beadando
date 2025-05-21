FROM python:3.11-slim

WORKDIR /app

COPY mlruns /app/mlruns
COPY mlartifacts /app/mlartifacts

RUN python -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip \
    && pip install mlflow==2.22.0 \
    && chmod -R 777 /app/mlruns \
    && chmod -R 777 /app/mlartifacts

ENV MLFLOW_TRACKING_URI="file:/app/mlruns"
ENV PATH="/app/venv/bin:$PATH"

CMD ["bash", "-c", "mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlartifacts"]