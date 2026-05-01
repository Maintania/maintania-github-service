from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base


class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)

    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)

    issue_number = Column(Integer, nullable=False)

    # 👇 Versioning
    version = Column(Integer, default=1)

    title = Column(String, nullable=False)
    body = Column(Text)
    issue_url = Column(String)

    comments = Column(Text)

    # AI outputs
    classification = Column(JSON)
    similar_fixes = Column(JSON)
    repo_context = Column(JSON)
    analysis = Column(JSON)

    # 👇 Tracking fields
    fetch_time_ms = Column(Float)   # how long API took
    is_latest = Column(Integer, default=1)  # 1 = latest, 0 = old snapshot

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    repository = relationship("Repository", back_populates="issues")