# Maintania GitHub Service

Before starting, make sure you have:

- Docker installed
- Docker Compose available through `docker compose`

## Environment Variables

The application reads configuration from environment variables. The Docker Compose file already provides:

- `DATABASE_URL=postgresql://user:password@postgres:5432/db`
- `QDRANT_URL=http://qdrant:6333`

## First-Time Setup

From the folder that contains `docker-compose.yml`, run:

```bash
docker compose up --build
```

## Everyday Start Commands

```bash
docker compose up -d
```

## After all this step run the pyhton project

## go into you rroot folder

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```


