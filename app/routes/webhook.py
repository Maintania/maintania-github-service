from fastapi import APIRouter, Request

from app.services.github_client import comment_on_issue

router = APIRouter()

@router.post("/")
async def github_webhook(request: Request):
    payload = await request.json()

    if payload.get("action") == "opened":

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



