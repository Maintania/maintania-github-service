from fastapi import APIRouter, Depends
import requests
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse, RedirectResponse

from app.core.config import settings
from app.core.security import create_access_token
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User

from urllib.parse import urlencode


router = APIRouter()


@router.get("/github/login")
def login():
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "scope": "read:user repo",
        "redirect_uri": settings.GITHUB_REDIRECT_URI
    }

    url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    return RedirectResponse(url)

@router.post("/logout")
def logout():

    response = JSONResponse({"message": "Logged out"})

    response.delete_cookie(
        key="session",
        httponly=True,
        samesite="lax"
    )

    return response



@router.get("/github/callback")
async def callback(code: str, db: Session= Depends(get_db)):

    token_res = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code
        }
    ).json()

    access_token = token_res["access_token"]

    user_res = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()

    github_id = str(user_res["id"])

    user = db.query(User).filter(User.github_id == github_id).first()

    if not user:
        user = User(
            github_id=github_id,
            username=user_res.get("login"),
            name=user_res.get("name"),
            avatar_url=user_res.get("avatar_url")
        )
        db.add(user)

    db.commit()

    token = create_access_token({"user_id": user.id})

    response = RedirectResponse("http://localhost:3000/dashboard")
    response.set_cookie("session", token, httponly=True)

    return response


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user