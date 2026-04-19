from sqlalchemy.orm import Session
from app.models.token_blacklist import TokenBlacklist


class TokenRepository:

    def add_to_blacklist(self, db: Session, jti: str, user_id: int, expires_at):
        token = TokenBlacklist(
            token_jti=jti,
            user_id=user_id,
            expires_at=expires_at
        )
        db.add(token)
        db.commit()

    def is_blacklisted(self, db: Session, jti: str):
        return db.query(TokenBlacklist).filter(
            TokenBlacklist.token_jti == jti
        ).first() is not None