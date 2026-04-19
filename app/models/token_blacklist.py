from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

from app.db.base import Base


class TokenBlacklist(Base):
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    token_jti = Column(String, index=True, unique=True)
    user_id = Column(Integer, index=True)

    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)