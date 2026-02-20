from sqlalchemy import Column, DateTime, Integer, String, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)

    github_repo_id = Column(String, unique=True, index=True)

    name = Column(String)
    full_name = Column(String)
    private = Column(Boolean)

    # Foreign key linking to installation
    installation_id = Column(Integer, ForeignKey("installations.id"))
    created_at = Column(DateTime(timezone=True),server_default=func.now(),nullable=False)
    updated_at = Column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now(),nullable=False)

    # Relationship back to installation
    installation = relationship("Installation", back_populates="repositories")
