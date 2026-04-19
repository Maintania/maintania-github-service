from sqlalchemy.orm import Session
from app.models.plan import Plan


class PlanRepository:

    def get_all(self, db: Session):
        return db.query(Plan).filter(Plan.is_active == True).all()

    def get_by_id(self, db: Session, plan_id: int):
        return db.query(Plan).filter(Plan.id == plan_id).first()

    def get_by_code(self, db: Session, code: str):
        return db.query(Plan).filter(
            Plan.code == code,
            Plan.is_active == True
        ).first()