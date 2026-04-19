import uuid
from datetime import datetime, timedelta
from jose import jwt

from app.core.config import settings
from app.repositories.token_repo import TokenRepository


class TokenService:

    def __init__(self):
        self.repo = TokenRepository()

    def create_access_token(self, data: dict):
        to_encode = data.copy()

        expire = datetime.utcnow() + timedelta(minutes=60)
        jti = str(uuid.uuid4())

        to_encode.update({
            "exp": expire,
            "jti": jti
        })

        token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm="HS256")

        return token, jti, expire

    def verify_token(self, token: str):
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload

    def blacklist_token(self, db, jti: str, user_id: int, expires_at):
        self.repo.add_to_blacklist(db, jti, user_id, expires_at)

    def is_token_blacklisted(self, db, jti: str):
        return self.repo.is_blacklisted(db, jti)