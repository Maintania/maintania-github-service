from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from app.core.config import settings
from app.db.session import get_db
from app.dependencies.auth import get_current_user
from app.models.installation import Installation
from app.models.repository import *
from app.services.github.github_client import *
from app.services.ai.issues import *
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
import os



client = QdrantClient(
    url=os.getenv("Qdrant_URL"), 
    api_key=os.getenv("Qdrant_Api_Key"),
)

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
    print("payload",payload)
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

            db.query(Repository).filter(
                Repository.installation_id == installation_id
            ).delete()
            
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
    issue_title: str
    issue_body: Optional[str] = ""


@router.post("/analyze-issue")
def test_maintania_pipeline(payload: AnalyzeIssuePayload):

    # ----------------------------
    # Phase 1 — Issue Classification
    # ----------------------------
    classification = classify_issue(payload.issue_title, payload.issue_body)

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

    # ----------------------------
    # Phase 3 — Retrieve Relevant Code From Vector DB
    # ----------------------------

    query = f"""
    
    Bug Report
    Title:
    {payload.issue_title}

    Description:
    {payload.issue_body}
    
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
        payload.issue_title,
        payload.issue_body,
        repo_context,
        None
    )

    return {
        "label":classification,
        "issue_number": payload.issue_number,
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