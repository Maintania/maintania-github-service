from fastapi import APIRouter, Depends
from app.repositories.plan_repo import PlanRepository
from app.db.session import get_db

router = APIRouter()

@router.get("/")
def get_plans(db=Depends(get_db)):
    repo = PlanRepository()
    return repo.get_all(db)