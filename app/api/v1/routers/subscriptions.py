from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.services.billing.subscription_service import SubscriptionService
from app.repositories.plan_repo import PlanRepository
from app.models.user import User

router = APIRouter()

subscription_service = SubscriptionService()
plan_repo = PlanRepository()


# ✅ CREATE SUBSCRIPTION
@router.post("/")
def create_subscription(
    payload: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    plan_code = payload.get("plan_code")

    if not plan_code:
        raise HTTPException(status_code=400, detail="plan_code required")

    plan = plan_repo.get_by_code(db, plan_code)

    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    existing = subscription_service.get_user_subscription(db, user.id)
    if existing:
        raise HTTPException(status_code=400, detail="Subscription already exists")

    subscription = subscription_service.create_subscription(
        db,
        user_id=user.id,
        plan_id=plan.id
    )

    return subscription


# ✅ GET CURRENT USER SUBSCRIPTION
@router.get("/me")
def get_my_subscription(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    subscription = subscription_service.get_user_subscription(db, user.id)

    if not subscription:
        raise HTTPException(status_code=404, detail="No subscription found")

    return subscription