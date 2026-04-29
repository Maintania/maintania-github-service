from fastapi import APIRouter, Depends, HTTPException, Request
import requests
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse, RedirectResponse
from urllib.parse import urlencode
from datetime import datetime

from app.services.auth.token_service import TokenService
from app.core.config import settings
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User

router = APIRouter()


# 🔹 GITHUB LOGIN
@router.get("/github/login")
def login():
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "scope": "read:user repo",
        "redirect_uri": settings.GITHUB_REDIRECT_URI
    }

    url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    return RedirectResponse(url)


# 🔹 GITHUB CALLBACK
@router.get("/github/callback")
async def callback(code: str, db: Session = Depends(get_db)):

    token_res = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code
        }
    ).json()

    # 🔴 Handle OAuth failure
    if "access_token" not in token_res:
        return JSONResponse(
            {"error": "OAuth failed", "details": token_res},
            status_code=400
        )

    access_token = token_res["access_token"]

    user_res = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()

    github_id = str(user_res["id"])

    user = db.query(User).filter(User.github_id == github_id,is_deleted=False).first()

    if not user:
        user = User(
            github_id=github_id,
            username=user_res.get("login"),
            name=user_res.get("name"),
            avatar_url=user_res.get("avatar_url"),
            is_active=True,
            is_verified=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # 🔹 Generate JWT with jti
    token_service = TokenService()
    token, jti, expire = token_service.create_access_token({
        "user_id": user.id
    })

    response = RedirectResponse(f"{settings.FRONTEND_URL}/repositeries")

    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        secure=False,       # 🔁 change to True in production
        samesite="lax",     # 🔁 change to "none" in production
        path="/",
        max_age=60 * 60 * 24
    )

    return response


# 🔹 LOGOUT
@router.post("/logout")
def logout(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    token_service = TokenService()

    token = request.cookies.get("session")

    if not token:
        raise HTTPException(status_code=401, detail="No token found")

    payload = token_service.verify_token(token)

    token_service.blacklist_token(
        db=db,
        jti=payload["jti"],
        user_id=user.id,
        expires_at=datetime.fromtimestamp(payload["exp"])
    )

    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("session")

    return response


# 🔹 CURRENT USER
@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user