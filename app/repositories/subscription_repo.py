from sqlalchemy.orm import Session
from app.models.subscription import Subscription

class SubscriptionRepository:

    def get_by_user(self, db: Session, user_id: int):
        return db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active"
        ).first()

    def create(self, db: Session, data: dict):
        sub = Subscription(**data)
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub