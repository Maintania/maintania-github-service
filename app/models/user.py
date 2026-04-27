from sqlalchemy import Column, DateTime, Integer, String, func, Boolean
from sqlalchemy.orm import relationship
from app.db.base import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(String, unique=True, index=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    avatar_url = Column(String)
    created_at = Column(DateTime(timezone=True),server_default=func.now(),nullable=False)
    updated_at = Column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now(),nullable=False)
    installations = relationship("Installation", back_populates="user")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False)
    

