FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip wheel \
    --no-cache-dir \
    --timeout 1000 \
    --retries 20 \
    -i https://pypi.org/simple \
    --wheel-dir /wheels \
    -r requirements.txt


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]