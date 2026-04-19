from sqlalchemy import Column, Integer, ForeignKey, String, DateTime, Boolean
from datetime import datetime

from app.db.base import Base

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("plans.id"))

    status = Column(String, default="active")  # active, cancelled, expired

    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)

    cancel_at_period_end = Column(Boolean, default=False)