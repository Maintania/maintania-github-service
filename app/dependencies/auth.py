from fastapi import Depends, HTTPException, Request
import jwt
from sqlalchemy.orm import Session


from app.db.session import get_db
from app.models.user import User
from app.core.config import settings

ALGORITHM = "HS256"


def get_current_user(request: Request, db: Session = Depends(get_db)):

    token = request.cookies.get("session")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

    except :
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


