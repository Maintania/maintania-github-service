from app.repositories.user_repo import UserRepository


class UserService:

    def __init__(self):
        self.repo = UserRepository()

    def get_user(self, db, user_id: int):
        return db.query(self.repo.model).filter(
            self.repo.model.id == user_id
        ).first()

    def update_profile(self, db, user, data: dict):
        for key, value in data.items():
            setattr(user, key, value)

        db.commit()
        db.refresh(user)

        return user

    def deactivate_user(self, db, user):
        user.is_active = False
        db.commit()
        return user

    def check_user_active(self, user):
        if not user.is_active:
            raise Exception("User account is deactivated")