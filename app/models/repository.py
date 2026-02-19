from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
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

    # Relationship back to installation
    installation = relationship("Installation", back_populates="repositories")
