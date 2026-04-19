FROM python:3.11-slim AS builder

WORKDIR /app


RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip wheel \
    --no-cache-dir \
    --timeout 1000 \
    --retries 20 \
    -i https://pypi.org/simple \
    --wheel-dir /wheels \
    -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

# runtime deps only (lighter)
RUN apt-get update && apt-get install -y \
    libpq5 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]