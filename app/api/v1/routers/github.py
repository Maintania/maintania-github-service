import hashlib
import hmac
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from app.core.config import settings
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.installation import Installation
from app.models.repository import *
from app.services.github.github_client import *
from app.services.ai.issues_copy import *
from app.services.repo.repository_service import sync_repositories  
import jwt
import time
import requests
from typing import Optional
from pydantic import BaseModel,Field
from app.services.ai.labelling import *
from app.services.repo.repo_cloner import *
from app.services.ai.issue_solver import *
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.services.repo.incremental_indexer import *
from app.db.session import SessionLocal
from fastapi import BackgroundTasks
import os



client = QdrantClient(
    url=os.getenv("Qdrant_URL"), 
    api_key=os.getenv("Qdrant_Api_Key"),
)

router = APIRouter()

def verify_signature(payload_body: bytes, signature: str, secret: str):
    if not signature:
        return False

    mac = hmac.new(secret.encode(), payload_body, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()

    return hmac.compare_digest(expected, signature)

def fetch_repos(token: str):
    url = "https://api.github.com/installation/repositories"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise Exception(f"GitHub API error: {res.text}")

    data = res.json()
    return data.get("repositories", [])


def upsert_repo(db, installation_id, repo):
    full_name = repo["full_name"]

    existing = db.query(Repository).filter(
        Repository.full_name == full_name,
        Repository.installation_id == installation_id
    ).first()

    if existing:
        existing.name = repo["name"]
        existing.private = repo["private"]
    else:
        db.add(Repository(
            installation_id=installation_id,
            name=repo["name"],
            full_name=full_name,
            private=repo["private"]
        ))


def sync_installation_repos(installation_db_id: int):
    db = SessionLocal()

    try:
        installation = db.query(Installation).filter(
            Installation.id == installation_db_id
        ).first()

        if not installation:
            return

        token = get_installation_token(installation.installation_id)
        repos = fetch_repos(token)

        incoming_full_names = set()

        for repo in repos:
            full_name = repo["full_name"]
            incoming_full_names.add(full_name)

            upsert_repo(db, installation.id, repo)

        if incoming_full_names:
            db.query(Repository).filter(
                Repository.installation_id == installation.id,
                ~Repository.full_name.in_(incoming_full_names)
            ).delete(synchronize_session=False)

        db.commit()

    finally:
        db.close()

# only source of truth if the insatllation happend 

@router.post("/webhook")
async def github_webhook(request: Request,background_tasks: BackgroundTasks, db: Session = Depends(get_db)):

    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_signature(raw_body, signature, settings.GITHUB_WEBHOOK_SECRET):
        raise HTTPException(401, "Invalid signature")

    payload = await request.json()
    event = request.headers.get("X-GitHub-Event")

    installation_data = payload.get("installation")
    installation_id = str(installation_data["id"]) if installation_data else None


    # INSTALLATION CREATED

    if event == "installation" and payload["action"] == "created":
        installation = db.query(Installation).filter(
            Installation.installation_id == installation_id
        ).first()

        if not installation:
            installation = Installation(
                installation_id=installation_id,
                account_login=installation_data["account"]["login"],
                account_type=installation_data["account"]["type"]
            )
            db.add(installation)
            db.commit()
            db.refresh(installation)

       # async instead of blocking
        background_tasks.add_task(sync_installation_repos, installation.id)

  
    # INSTALLATION DELETED
    elif event == "installation" and payload["action"] == "deleted":

        installation = db.query(Installation).filter(
            Installation.installation_id == installation_id
        ).first()

        if installation:
            db.query(Repository).filter(
                Repository.installation_id == installation.id
            ).delete()

            db.delete(installation)
            db.commit()

    # REPO ADDED/REMOVED
    elif event == "installation_repositories":

        installation = db.query(Installation).filter(
          Installation.installation_id == installation_id
        ).first()

        if installation:
            background_tasks.add_task(sync_installation_repos, installation.id)
            return {"ok": True}

        if payload["action"] == "added":
            for repo in payload["repositories_added"]:
                upsert_repo(db, installation.id, repo)

        elif payload["action"] == "removed":
            for repo in payload["repositories_removed"]:
                db.query(Repository).filter(
                    Repository.full_name == repo["full_name"]
                ).delete()

            db.commit()

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

    return installation.repositories


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



class RepoIndexRequest(BaseModel):
    owner: str
    repo: str
    branch: str | None = None
    installation_id: int
    type: str | None = None




@router.post("/index-repository")
def index_repository(request: RepoIndexRequest):

    engine = RepoIntelligenceEngine()

    owner = request.owner
    repo = request.repo

    if request.type == "reset":
        engine.reset_collection()
        return {
            "status": "success",
            "message": "Collection reset"
        }

    if request.type == "delete":

        if request.branch is None:

            engine.delete_repo(
                owner=request.owner,
                repo=request.repo
            )
            engine.delete_state(
                owner=request.owner,
                repo=request.repo
            )

            return {
                "status": "success",
                "repo": f"{request.owner}/{request.repo}"
            }

        else:

            engine.delete_repo(
                owner=request.owner,
                repo=request.repo,
                branch=request.branch
            )
            engine.delete_repo_state(
                owner=request.owner,
                repo=request.repo,
                branch=request.branch
            )

            return {
                "status": "success",
                "branch": request.branch,
                "repo": f"{request.owner}/{request.repo}"
            }

    try:

        start_time = time.time()
        target_branch = request.branch
        if not request.branch:
            branch_name = engine.resolve_branch(
                owner=request.owner,
                repo=request.repo,
                installation_id=request.installation_id,
                branch=request.branch
            )
        else:
            branch_name = target_branch
            
        print("Branch:", branch_name)
            
            
        existing_state = engine.get_repo_state(
            owner=request.owner,
            repo=request.repo,
            branch=branch_name
        )

        # -----------------------------------
        # IF EXISTS → CALL SYNC (INCREMENTAL)
        # -----------------------------------
        if existing_state and existing_state.get("last_commit"):

            print("Repo already indexed → running incremental sync")

            return sync_repo(SyncRepoPayload(
                installation_id=request.installation_id,
                owner=request.owner,
                repo=request.repo,
                branch=branch_name
            ))
        
        # Clone repository
        branch = engine.clone_repo(
            owner=request.owner,
            repo=request.repo,
            installation_id=request.installation_id,
            branch=branch_name
        )

        engine.upsert_repo_state(
            owner=request.owner,
            repo=request.repo,
            branch=branch,
            data={
                "status": "indexing",
                "last_update_type": "full",
                "error": None
            }
        )

        # Process and store embeddings
        stats, success = engine.process_repository(
            owner=request.owner,
            repo=request.repo,
            branch=branch
        )
        repo_obj = Repo(engine.repo_root)
        current_commit = repo_obj.head.commit.hexsha

        engine.upsert_repo_state(
            owner=request.owner,
            repo=request.repo,
            branch=branch,
            data={
                "last_commit": current_commit,
                "total_files": stats["total_files"],
                "total_chunks": stats["total_chunks"],
                "languages": stats["languages"],
                "last_indexed_at": engine._utc_now_iso(),
                "last_index_duration_sec": stats["duration_sec"],
                "status": "ready",
                "last_update_type": "full",
                "error": None
            }
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="No supported files found in repository"
            )

        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "repository": f"{request.owner}/{request.repo}",
            "branch": branch,
            "chunks_indexed": stats["total_chunks"],
            "index_time_seconds": round(time.time() - start_time, 2)
        }

    except Exception as e:

        if 'branch' in locals():
            engine.upsert_repo_state(
                owner=request.owner,
                repo=request.repo,
                branch=branch,
                data={
                    "status": "failed",
                    "last_indexed_at": engine._utc_now_iso(),
                    "last_index_duration_sec": round(time.time() - start_time, 2),
                    "last_update_type": "full",
                    "error": str(e)
                }
            )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        engine.cleanup()


class AnalyzeIssuePayload(BaseModel):
    installation_id: int
    owner: str
    repo: str
    issue_number: int
    # issue_title: str
    # issue_body: Optional[str] = ""

@router.post("/analyze-issue")
def test_maintania_pipeline(payload: AnalyzeIssuePayload):

    # ----------------------------
    # Phase 0 — Fetch Issue From GitHub (SOURCE OF TRUTH)
    # ----------------------------
    issue = github_get_issue_details(
        installation_id=payload.installation_id,
        owner=payload.owner,
        repo=payload.repo,
        issue_number=payload.issue_number
    )

    issue_title = issue["title"]
    issue_body = issue["body"]

    # OPTIONAL: fetch comments (recommended)
    comments = github_get_issue_comments(
        installation_id=payload.installation_id,
        owner=payload.owner,
        repo=payload.repo,
        issue_number=payload.issue_number
    )

    # Merge comments into context (limit to avoid token explosion)
    comments_text = "\n\n".join(
        [c["body"] for c in comments[:5]]
    )

    # enriched_body = f"""
    # {issue_body}

    # --- Comments ---
    # {comments_text}
    # """
    
    enriched_body = f"""
    {issue_body}
    """

    # ----------------------------
    # Phase 1 — Issue Classification
    # ----------------------------
    classification = classify_issue(issue_title, enriched_body)

    # ----------------------------
    # Phase 2 — Find Similar Fixes
    # ----------------------------
    results = maintania_find_similar_fixes(
        installation_id=payload.installation_id,
        title=issue_title,
        body=enriched_body,
        owner=payload.owner,
        repo=payload.repo,
        issue_number=payload.issue_number,
        top_k=10
    )

    # ----------------------------
    # Phase 3 — Retrieve Relevant Code From Vector DB
    # ----------------------------
    query = f"""
    Bug Report
    Title:
    {issue_title}

    Description:
    {enriched_body}

    Find code responsible for this issue.
    """

    query_vector = embed([query])[0]

    repo_full_name = f"{payload.owner}/{payload.repo}"

    search_results = client.query_points(
        collection_name="repo_code_embeddings",
        query=query_vector,
        limit=10,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="repo",
                    match=MatchValue(value=repo_full_name)
                )
            ]
        )
    )

    repo_context = []

    for r in search_results.points:
        repo_context.append({
            "file": r.payload.get("file"),
            "code": r.payload.get("code"),
            "score": r.score
        })

    # ----------------------------
    # Phase 4 — Root Cause Analysis
    # ----------------------------
    root_engine = RootCauseEngine()

    analysis = root_engine.analyze(
        issue_title,
        enriched_body,
        repo_context,
        None
    )

    return {
        "label": classification,
        "issue_number": payload.issue_number,
        "issue_title": issue_title,
        "issue_url": issue["html_url"],
        "similar_fixes": results,
        "repo_context": repo_context,
        "analysis": analysis
    }  


# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class SyncRepoPayload(BaseModel):
    installation_id: int
    owner: str
    repo: str

    # optional
    branch: str | None = None
    pr_number: int | None = None
    base_branch: str | None = "main"


# -----------------------------
# HELPER: UPDATE EXISTING CLONE
# -----------------------------
def update_repo(repo_path, branch):
    repo = Repo(repo_path)

    repo.remotes.origin.fetch()

    repo.git.checkout(branch)
    repo.git.reset("--hard", f"origin/{branch}")


# -----------------------------
# HELPER: CHECKOUT PR
# -----------------------------
def checkout_pr(repo_path, pr_number):
    repo = Repo(repo_path)

    repo.remotes.origin.fetch(
        f"pull/{pr_number}/head:pr_{pr_number}"
    )

    repo.git.checkout(f"pr_{pr_number}")


# -----------------------------
# MAIN API
# -----------------------------
@router.post("/sync-repo")
def sync_repo(payload: SyncRepoPayload):

    engine = RepoIntelligenceEngine()

    # -----------------------------
    # STEP 1: CLONE (if not exists)
    # -----------------------------
    branch = payload.branch

    if not branch:
        branch = "main"

    branch = engine.clone_repo(
        payload.owner,
        payload.repo,
        payload.installation_id,
        payload.branch
    )

    # -----------------------------
    # STEP 2: HANDLE PR vs PUSH
    # -----------------------------
    repo_path = engine.repo_root

    if payload.pr_number:
        print(f"Processing PR #{payload.pr_number}")

        checkout_pr(repo_path, payload.pr_number)

    else:
        print(f"Processing branch: {branch}")

        update_repo(repo_path, branch)

    # -----------------------------
    # STEP 3: INCREMENTAL INDEXING
    # -----------------------------
    indexer = IncrementalIndexer(engine)

    result = indexer.run(
        payload.owner,
        payload.repo,
        branch
    )

    # -----------------------------
    # STEP 4: FULL INDEX (FIRST TIME)
    # -----------------------------
    if result == "FULL_REINDEX":

        sync_start = time.time()
        engine.upsert_repo_state(
            payload.owner,
            payload.repo,
            branch,
            {
                "status": "indexing",
                "last_update_type": "full",
                "error": None
            }
        )

        try:
            stats, _ = engine.process_repository(
                payload.owner,
                payload.repo,
                branch
            )
            repo_obj = Repo(engine.repo_root)
            current_commit = repo_obj.head.commit.hexsha

            update_type = "pr" if payload.pr_number else "full"
            engine.upsert_repo_state(
                payload.owner,
                payload.repo,
                branch,
                {
                    "last_commit": current_commit,
                    "total_files": stats["total_files"],
                    "total_chunks": stats["total_chunks"],
                    "languages": stats["languages"],
                    "last_indexed_at": engine._utc_now_iso(),
                    "last_index_duration_sec": round(time.time() - sync_start, 2),
                    "status": "ready",
                    "last_update_type": update_type,
                    "error": None
                }
            )
        except Exception as e:
            engine.upsert_repo_state(
                payload.owner,
                payload.repo,
                branch,
                {
                    "status": "failed",
                    "last_indexed_at": engine._utc_now_iso(),
                    "last_index_duration_sec": round(time.time() - sync_start, 2),
                    "last_update_type": "pr" if payload.pr_number else "full",
                    "error": str(e)
                }
            )
            raise

    elif result in ("UPDATED", "NO_CHANGE") and payload.pr_number:
        # mark update type for PR sync without recomputing full stats
        state = engine.get_repo_state(payload.owner, payload.repo, branch) or {}
        engine.upsert_repo_state(
            payload.owner,
            payload.repo,
            branch,
            {
                "last_commit": state.get("last_commit"),
                "last_indexed_at": engine._utc_now_iso(),
                "last_update_type": "pr",
                "status": "ready",
                "error": None
            }
        )

    # -----------------------------
    # CLEANUP
    # -----------------------------
    engine.cleanup()

    return {
        "status": result,
        "repo": f"{payload.owner}/{payload.repo}",
        "branch": branch,
        "pr": payload.pr_number
    }


class RepoStatsRequest(BaseModel):
    owner: str = Field(..., description="GitHub owner (required)")
    repo: Optional[str] = None
    branch: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


@router.get("/repo-stats")
def repo_stats(request: RepoStatsRequest):
    engine = RepoIntelligenceEngine()
    engine.create_state_collection()
    owner = request.owner
    repo = request.repo
    branch = request.branch
    limit = request.limit
    offset = request.offset
    # -----------------------------------
    # VALIDATION
    # -----------------------------------
    if not owner:
        raise HTTPException(
            status_code=400,
            detail="owner is required"
        )

    # -----------------------------------
    # CASE 3: owner + repo + branch
    # -----------------------------------
    if owner and repo and branch:
        record = engine.get_repo_state(owner, repo, branch)

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Repository state not found"
            )

        return {
            "count": 1,
            "items": [record],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "next_offset": None
            },
            "aggregation": {
                "total_repos": 1,
                "total_chunks": int(record.get("total_chunks") or 0)
            }
        }

    # -----------------------------------
    # FETCH ALL (will filter below)
    # -----------------------------------
    records = []
    next_offset = None
    cursor = None

    while True:
        points, cursor = engine.qdrant.scroll(
            collection_name=engine.state_collection,
            limit=200,
            offset=cursor,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}

            repo_name = payload.get("repo")  # format: owner/repo

            if not repo_name:
                continue

            try:
                payload_owner, payload_repo = repo_name.split("/")
            except ValueError:
                continue

            # -----------------------------------
            # CASE 1: owner only
            # -----------------------------------
            if owner and not repo:
                if payload_owner != owner:
                    continue

            # -----------------------------------
            # CASE 2: owner + repo
            # -----------------------------------
            elif owner and repo and not branch:
                if payload_owner != owner or payload_repo != repo:
                    continue

            records.append(payload)

        if cursor is None:
            break

    # -----------------------------------
    # PAGINATION + AGGREGATION
    # -----------------------------------
    total_repos = len(records)
    total_chunks = sum(int(item.get("total_chunks") or 0) for item in records)

    page_items = records[offset:offset + limit]

    if offset + limit < total_repos:
        next_offset = offset + limit

    return {
        "count": len(page_items),
        "items": page_items,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset
        },
        "aggregation": {
            "total_repos": total_repos,
            "total_chunks": total_chunks
        }
    }