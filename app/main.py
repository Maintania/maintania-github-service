from fastapi import FastAPI
from app.routes import  auth, github, health
from app.db.init_db import init_db
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Maintania GitHub Automation")

init_db()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health")
app.include_router(github.router, prefix="/github")
app.include_router(auth.router, prefix="/auth")

