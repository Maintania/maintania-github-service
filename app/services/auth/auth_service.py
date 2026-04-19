from app.repositories.user_repo import UserRepository
from app.core.security import create_access_token

class AuthService:

    def __init__(self):
        self.user_repo = UserRepository()

    def login_with_github(self, db, github_user):
        github_id = str(github_user["id"])

        user = self.user_repo.get_by_github_id(db, github_id)

        if not user:
            user = self.user_repo.create(db, {
                "github_id": github_id,
                "username": github_user.get("login"),
                "name": github_user.get("name"),
                "avatar_url": github_user.get("avatar_url")
            })

        token = create_access_token({"user_id": user.id})
        return user, token