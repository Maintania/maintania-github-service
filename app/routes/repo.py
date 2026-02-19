from fastapi import APIRouter

from app.services.github_client import get_installation_repos

router = APIRouter()

router.get("/github/repos")
async def list_repos():
 
    installation_id = INSTALLATIONS.get("current")

    if not installation_id:
        return {"error": "No installation found"}

    repos = get_installation_repos(installation_id)

    return repos