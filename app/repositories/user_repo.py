from sqlalchemy.orm import Session
from app.models.user import User

class UserRepository:
    model = User
    def get_by_github_id(self, db: Session, github_id: str):
        return db.query(User).filter(User.github_id == github_id).first()

    def create(self, db: Session, data: dict):
        user = User(**data)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user 