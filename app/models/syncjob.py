from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.db.base import Base


class SyncJob(Base):
    __tablename__ = "sync_jobs"

    id = Column(Integer, primary_key=True)

    installation_id = Column(Integer, index=True)
    repo_full_name = Column(String, index=True)

    status = Column(String)  
    # PENDING | RUNNING | SUCCESS | FAILED

    progress = Column(Integer, default=0)  # 0–100

    attempt = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)