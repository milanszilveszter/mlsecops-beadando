FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY constants.py .
COPY MLModel.py .

RUN python -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

ENV PATH="/app/venv/bin:$PATH"

CMD ["python", "app.py"]