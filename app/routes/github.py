from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from app.core.config import settings
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.installation import Installation
from app.services.github_client import *
from app.services.Issues import *
from app.services.repository_service import sync_repositories
import jwt
import time
import requests
from typing import Optional
from pydantic import BaseModel
from app.services.labelling import *
from app.services.repo_cloner import *
from app.services.issue_solver import *




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
    print(payload)
    if event == "installation":

        action = payload.get("action")
        installation = payload.get("installation")

        installation_id = str(installation["id"])
        account_login = installation["account"]["login"]
        account_type = installation["account"]["type"]

        if action == "created":

            existing = db.query(Installation).filter(
                Installation.installation_id == installation_id
            ).first()

            if not existing:
                new_installation = Installation(
                    installation_id=installation_id,
                    account_login=account_login,
                    account_type=account_type
                )
                db.add(new_installation)

        elif action == "deleted":

            db.query(Installation).filter(
                Installation.installation_id == installation_id
            ).delete()

        elif action == "suspend":
            print("Installation suspended:", installation_id)

        elif action == "unsuspend":
            print("Installation unsuspended:", installation_id)

        db.commit()

    elif event == "installation_repositories":

        action = payload.get("action")
        installation_id = str(payload["installation"]["id"])

        print("Repo change detected:", action, installation_id)
    elif event == "issues":

        action = payload.get("action")

        if action == "opened":

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


@router.get("/issues/{installation_id}/{owner}/{repo}/")
def call_issues(
    installation_id: int,
    owner: str,
    repo: str,
    q: str = "",
    state: str = "all",
    limit: int = 100
):
    """
    Fetch issues from repo.
    If q is provided → keyword filter.
    """

    token = get_installation_token(installation_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    issues = []
    page = 1

    while len(issues) < limit:

        res = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            headers=headers,
            params={
                "state": state,
                "per_page": 100,
                "page": page
            },
            timeout=20
        )

        res.raise_for_status()

        data = res.json()
        if not data:
            break

        if q:
            q_lower = q.lower()
            data = [
                i for i in data
                if q_lower in (i.get("title", "") + " " + (i.get("body") or "")).lower()
            ]

        issues.extend(data)
        page += 1

    return issues[:limit]

class AnalyzeIssuePayload(BaseModel):
    installation_id: int
    owner: str
    repo: str
    issue_number: int
    issue_title: str
    issue_body: Optional[str] = ""

# @router.post("/analyze-issue")
# def test_maintania_pipeline(
#     installation_id: int,
#     owner: str,
#     repo: str,
#     issue_number: int,
#     issue_title: str,
#     issue_body: Optional[str] = ""
# ):
#     """
#     Manual test endpoint for Maintania AI pipeline.
#     """

#     try:
#         # # 1. Post initial comment (same as webhook flow)
#         # comment_on_issue(
#         #     installation_id,
#         #     owner,
#         #     repo,
#         #     issue_number,
#         #     "Issue received 👀 Maintania will analyze this."
#         # )

#         # 2. Run similarity pipeline
#         results = maintania_find_similar_fixes(
#             installation_id=installation_id,
#             title=issue_title,
#             body=issue_body,
#             owner=owner,
#             repo=repo,
#             embed_fn=embed_fn,   # your embedding function
#             top_k=5
#         )

#         return {
#             "status": "success",
#             "issue": {
#                 "owner": owner,
#                 "repo": repo,
#                 "issue_number": issue_number,
#                 "title": issue_title,
#             },
#             "similar_fixes": results
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/analyze-issue")
def test_maintania_pipeline(payload: AnalyzeIssuePayload):
    """
    Manual test endpoint for Maintania AI pipeline.
    """
    # ----------------------------
    # Phase 1 — Issue Classification
    # ----------------------------

    # classification  = classify_issue(payload.issue_title, payload.issue_body)
    # print(classification)
    # ----------------------------
    # Phase 2 — Find Similar Fixes 
    # ----------------------------

    results = maintania_find_similar_fixes(
        installation_id=payload.installation_id,
        title=payload.issue_title,
        body=payload.issue_body,
        owner=payload.owner,
        repo=payload.repo,
        issue_number=payload.issue_number,
        top_k=10
    )

    print(results)
    # # ----------------------------
    # # Phase 3 — Repo Cloner and Finding the Relevent Code Blocks
    # # ----------------------------
    engine = RepoIntelligenceEngine()

    try:
        engine.clone_repo(
            owner=payload.owner,
            repo=payload.repo,
            installation_id=payload.installation_id
        )

        success = engine.process_repository()

        if not success:
            return {"error": "No valid source files found"}

        query = f"{payload.issue_title}\n{payload.issue_body}"

        repo_context = engine.search(query, top_k=5)

        file_tree = engine.file_tree
        
        
        root_engine = RootCauseEngine()
        
        analysis = root_engine.analyze(
            payload.issue_title,
            payload.issue_body,
            repo_context,
            file_tree
        )

    finally:
        engine.cleanup()


    return {
        "issue_number": payload.issue_number,
        # 'labels': classification,
        "similar_fixes": results,
        "repo_context": repo_context,
        "file_tree":file_tree,
        'analysis': analysis

    }
    
    
