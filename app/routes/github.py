from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from app.core.config import settings
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.installation import Installation
from app.services.github_client import comment_on_issue, get_installation_repos
from app.services.repository_service import sync_repositories

router = APIRouter()

@router.get("/setup")
def github_setup(installation_id: str, user = Depends(get_current_user), db: Session = Depends(get_db)):

    existing = db.query(Installation).filter(
        Installation.installation_id == installation_id,
        Installation.user_id == user.id
    ).first()

    if not existing:
        installation = Installation(
            installation_id=installation_id,
            user_id=user.id
        )
        db.add(installation)
        db.commit()
    sync_repositories(db, installation)
    return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard")

# get installation id from github callback query 
@router.post("/webhook")
async def github_webhook(request: Request, db: Session = Depends(get_db)):

    payload = await request.json()
    event = request.headers.get("X-GitHub-Event")

    if event == "installation":

        installation = payload["installation"]

        installation_id = str(installation["id"])
        account_login = installation["account"]["login"]
        account_type = installation["account"]["type"]

        existing = db.query(Installation).filter(
            Installation.installation_id == installation_id
        ).first()

        if existing:
            existing.account_login = account_login
            existing.account_type = account_type
        else:
            new_installation = Installation(
                installation_id=installation_id,
                account_login=account_login,
                account_type=account_type
            )
            db.add(new_installation)

        db.commit()

    elif event == "issues" and payload.get("action") == "opened":

        installation_id = payload["installation"]["id"]
        issue_number = payload["issue"]["number"]
        repo_name = payload["repository"]["name"]
        owner = payload["repository"]["owner"]["login"]

        comment_on_issue(
            installation_id,
            owner,
            repo_name,
            issue_number,
            "Issue received 👀 Maintania will analyze this."
        )

    return {"ok": True}


@router.get("/installations")
def list_installations(user = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Installation).filter(
        Installation.user_id == user.id
    ).all()


@router.get("/repos/{installation_id}")
def get_repos(
    installation_id: str,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    installation = db.query(Installation).filter(
        Installation.installation_id == installation_id,
        Installation.user_id == user.id
    ).first()

    if not installation:
        raise HTTPException(status_code=403, detail="Not allowed")

    return get_installation_repos(installation_id)




