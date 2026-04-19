from fastapi import APIRouter
from app.api.v1.routers import auth, users, plans, subscriptions, github, health

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(plans.router, prefix="/plans", tags=["plans"])
api_router.include_router(subscriptions.router, prefix="/subscriptions", tags=["subscriptions"])
api_router.include_router(github.router, prefix="/github", tags=["github"])
api_router.include_router(health.router, prefix="/health", tags=["health"])