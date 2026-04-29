from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.db.session import get_db
from app.models.user import User
from app.core.config import settings
from app.services.auth.token_service import TokenService

ALGORITHM = "HS256"

token_service = TokenService()


def get_current_user(request: Request, db: Session = Depends(get_db)):

    token = request.cookies.get("session")
    # token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjE3NzcxOTk0NzMsImp0aSI6Ijg1Mjk0NDhiLWRiNWMtNDM0Yi1hZDdhLTg2MWRlNmM4N2UzMiJ9.fbYYmIl_RSMoCXcORNY3XC_wsXWZvNJ59vuHJ2YtC54'

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("user_id")
        jti = payload.get("jti")

        if not user_id or not jti:
            raise HTTPException(status_code=401, detail="Invalid token")

        # 🔥 Check if token is revoked
        if token_service.is_token_blacklisted(db, jti):
            raise HTTPException(status_code=401, detail="Token revoked")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Fetch user
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # 🔥 Check active status
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")

    return user