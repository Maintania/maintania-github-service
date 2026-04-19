from fastapi import FastAPI
from app.api.v1.routers import auth, github
from app.core.config import settings
from app.api.v1.routers import  health
from app.db.init_db import init_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.api.v1.api import api_router

app = FastAPI(title="Maintania GitHub Automation")

init_db()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health")
app.include_router(github.router, prefix="/github")
app.include_router(auth.router, prefix="/auth")
app.include_router(api_router, prefix="/api/v1")



