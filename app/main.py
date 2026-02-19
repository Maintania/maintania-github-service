from fastapi import FastAPI
from app.routes import callback, health, repo, webhook
from app.db.init_db import init_db

app = FastAPI(title="Maintania GitHub Automation")
init_db()
app.include_router(health.router, prefix="/health")
app.include_router(webhook.router, prefix="/github/webhook")
app.include_router(repo.router, prefix="/github/repo")
app.include_router(callback.router, prefix="/github/callback")



