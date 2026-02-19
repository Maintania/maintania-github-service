from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class Installation(Base):
    __tablename__ = "installations"

    id = Column(Integer, primary_key=True, index=True)

    installation_id = Column(String, unique=True, index=True)

    account_login = Column(String)
    account_type = Column(String)

    # Foreign key linking to user
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationship back to user
    user = relationship("User", back_populates="installations")

    # Relationship to repositories
    repositories = relationship("Repository", back_populates="installation")
