from sqlalchemy.orm import Session
from app.models.repository import Repository
from app.models.installation import Installation
from app.services.github_client import get_installation_repos


def sync_repositories(db: Session, installation: Installation):

    data = get_installation_repos(installation.installation_id)

    repos = data.get("repositories", [])

    for repo in repos:

        existing = db.query(Repository).filter(
            Repository.github_repo_id == str(repo["id"])
        ).first()

        if existing:
            existing.name = repo["name"]
            existing.full_name = repo["full_name"]
            existing.private = repo["private"]
        else:
            new_repo = Repository(
                github_repo_id=str(repo["id"]),
                name=repo["name"],
                full_name=repo["full_name"],
                private=repo["private"],
                installation_id=installation.id
            )
            db.add(new_repo)

    db.commit()