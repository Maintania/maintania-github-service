# Maintania GitHub Service

This project runs a FastAPI service with PostgreSQL and Qdrant using Docker Compose.

## What This Starts

- `api`: FastAPI app exposed on `http://localhost:8000`
- `postgres`: PostgreSQL database exposed on `localhost:5432`
- `qdrant`: Qdrant vector database exposed on `http://localhost:6333`

## Prerequisites

Before starting, make sure you have:

- Docker installed
- Docker Compose available through `docker compose`

## Environment Variables

The application reads configuration from environment variables. The Docker Compose file already provides:

- `DATABASE_URL=postgresql://user:password@postgres:5432/db`
- `QDRANT_URL=http://qdrant:6333`

Depending on the features you use, you may also need to provide:

- `GITHUB_APP_ID`
- `GITHUB_APP_PRIVATE_KEY`
- `GITHUB_CLIENT_ID`
- `GITHUB_CLIENT_SECRET`
- `GITHUB_REDIRECT_URI`
- `WEBHOOK_SECRET`
- `MODEL_SERVICE_URL`
- `FRONTEND_URL`
- `JWT_SECRET_KEY`
- `GEMINI_API_KEY`

## First-Time Setup

From the folder that contains `docker-compose.yml`, run:

```bash
docker compose up --build
```

Use this the first time because Docker needs to:

- build the API image
- install Python dependencies from `requirements.txt`
- start PostgreSQL and Qdrant

## Everyday Start Commands

If the containers were already built and you just want to start the app again, run:

```bash
docker compose up -d
```

This will reuse the existing image and containers when nothing important changed.

## When You Need `--build`

Run with `--build` again only when you change something that affects the Docker image, for example:

- `Dockerfile`
- `requirements.txt`
- installed or removed Python packages

Command:

```bash
docker compose up -d --build
```

## Stop And Restart

Stop containers but keep their data:

```bash
docker compose stop
```

Start stopped containers again:

```bash
docker compose start
```

Stop and remove containers:

```bash
docker compose down
```

## View Logs

View logs for all services:

```bash
docker compose logs -f
```

View logs only for the API service:

```bash
docker compose logs -f api
```

## Open A Shell In The API Container

If `bash` is available:

```bash
docker compose exec api bash
```

If `bash` is not available:

```bash
docker compose exec api sh
```

## Verify Everything Is Running

After startup, you can check:

- FastAPI docs: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/health/`
- Versioned health endpoint: `http://localhost:8000/api/v1/health/`
- Qdrant: `http://localhost:6333`

## Quick Start Summary

1. Open a terminal in the project folder.
2. Run `docker compose up --build` the first time.
3. Wait for the containers to finish starting.
4. Open `http://localhost:8000/health/` to confirm the API is running.
5. Use `docker compose up -d` for normal restarts after that.
\

