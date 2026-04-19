from sqlalchemy import Column, Integer, String, Float, Boolean

from app.db.base import Base

class Plan(Base):
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)

    name = Column(String)
    description = Column(String)

    price = Column(Float)
    currency = Column(String, default="INR")
    interval = Column(String)  # monthly, yearly

    max_repos = Column(Integer, default=1)
    max_requests = Column(Integer, default=100)

    is_active = Column(Boolean, default=True)