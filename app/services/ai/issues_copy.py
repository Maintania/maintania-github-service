import re
import numpy as np
import requests
import os
import json
from typing import List, Dict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import login

from app.services.github.github_client import get_installation_token
from app.services.repo.repo_cloner import embed
from app.services.ai.llm_client import LLMClient
from google import genai


# ==========================
# INIT
# ==========================

login(token=os.getenv("HF_TOKEN"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

MAX_LLM_CHECK = 5

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

llm = LLMClient()

# ==========================
# EMBEDDING
# ==========================

def embed_texts_batch(texts: list[str]):
    if not texts:
        return []
    embeddings = embed(texts)
    return embeddings.tolist()


# ==========================
# SIMILARITY
# ==========================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom) if denom else 0.0


def jaccard(a: str, b: str):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# ==========================
# STACKTRACE
# ==========================

def detect_stacktrace(text: str):
    patterns = [
        r"traceback",
        r"stack trace",
        r"\bat\s+[a-zA-Z_]+\(",
        r"\.js:\d+",
        r"\.py:\d+",
        r"\.ts:\d+"
    ]
    return any(re.search(p, text.lower()) for p in patterns)


# ==========================
# CLEAN TEXT
# ==========================

def clean_issue_text(text: str):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return text.strip()


# ==========================
# KEYWORD EXTRACTION
# ==========================
def extract_signals(title, body):
    return {
        "functions": re.findall(r"\b\w+\.\w+\(\)", body),
        "files": re.findall(r"[\w/\\.-]+\.(ts|js|py)", body),
        "errors": re.findall(r"(Error|Exception|TypeError):.*", body),
        "stack_lines": re.findall(r"\.ts:\d+|\.js:\d+", body)
    }
    
    
def extract_keywords(title: str, body: str, k=8):

    body = clean_issue_text(body)
    text = f"{title} {body}".lower()

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_.:-]{2,}\b", text)
    stop = {
        "this","that","with","from","have","your","will","when",
        "what","which","there","about","where","would","could",
        "their","they","them","then","than","into","onto","using",
        "issue","error","problem","request","github"
    }

    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1

    return sorted(freq, key=freq.get, reverse=True)[:k]


# ==========================
# LLM KEYWORDS
# ==========================
def build_search_query(owner, repo, phrase, signals):
    parts = [f'"{phrase}"']

    if signals["functions"]:
        parts.append(f'"{signals["functions"][0]}"')

    if signals["errors"]:
        parts.append(f'"{signals["errors"][0][:50]}"')

    query = " OR ".join(parts)

    return f"""
repo:{owner}/{repo}
({query})
in:title,body
is:issue
is:closed
sort:updated-desc
"""

def clean_llm_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\[.*?\]", text, re.S)
    return json.loads(match.group(0)) if match else []


def build_signals_from_keywords(keywords):
    return {
        "functions": [k for k in keywords if "(" in k or "." in k],
        "errors": [k for k in keywords if "error" in k.lower()],
        "files": [k for k in keywords if "." in k and "/" in k],
        "keywords": keywords
    }
    

def llm_extract_keywords(title: str, body: str):
    if not GEMINI_API_KEY:
        return [], 0

    prompt = f"""
Extract 5 precise technical keywords from this issue.

Focus on:
- APIs
- functions
- errors
- frameworks

Title:
{title}

Body:
{body[:600]}

Return JSON list only.
"""
    # ---------------------------
    # Token Count
    # ---------------------------
    try:
        token_data = llm.count_tokens("gemini", MODEL_NAME, prompt)
        token_count = token_data["total_tokens"]
    except Exception:
        token_count = 0
    try:
        # ---------------------------
        # LLM Call
        # ---------------------------
        result_obj = llm.generate("gemini", MODEL_NAME, prompt)
        text = result_obj["text"]

        return clean_llm_json(text), token_count

    except:
        return [], token_count


# ==========================
# PHRASES
# ==========================

def keyword_phrases(keywords, max_phrases=6):
    phrases = []
    for k in (2, 3):
        for combo in combinations(keywords, k):
            phrases.append(" ".join(combo))
    return phrases[:max_phrases]


# ==========================
# GITHUB SEARCH
# ==========================

def search_phrase(url, headers, owner, repo, phrase, keywords, per_query):
    
    q = build_search_query(owner, repo, phrase, keywords)

    res = requests.get(
        url,
        headers=headers,
        params={
            "q": q,
            "per_page": per_query,
            "sort": "updated",
            "order": "desc"
        },
        timeout=30
    )

    if res.status_code != 200:
        return []

    return res.json().get("items", [])


def github_recent_keyword_search(
    installation_id: int,
    owner: str,
    repo: str,
    signals: dict,   # ✅ FIXED
    exclude_issue_number: int,
    per_query: int = 8
):

    token = get_installation_token(installation_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    url = "https://api.github.com/search/issues"
    results = {}
    keywords = signals.get("keywords", [])
    phrases = keyword_phrases(keywords)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                search_phrase,
                url,
                headers,
                owner,
                repo,
                p,
                signals,   # ✅ pass signals (NOT keywords)
                per_query
            )
            for p in phrases
        ]

        for future in as_completed(futures):
            try:
                items = future.result()
            except Exception as e:
                print("Search error:", e)
                continue

            for item in items:
                if item["number"] == exclude_issue_number:
                    continue

                results[item["id"]] = {
                    "number": item["number"],
                    "title": item["title"],
                    "body": item.get("body") or "",
                    "html_url": item["html_url"]
                }

    return list({c["number"]: c for c in results.values()}.values())


# ==========================
# LLM RERANK
# ==========================

def llm_rerank(query_title, query_body, candidates):

    if not GEMINI_API_KEY or not candidates:
        return candidates, 0

    prompt = f"""
Rank these issues by similarity to the query.

Query:
{query_title}
{query_body[:400]}

Criteria:
Rank issues by likelihood of SAME ROOT CAUSE.

Strong signals:
1. Same function/API name (exact match is very strong)
2. Same error message or stacktrace pattern
3. Same file/module mentioned
4. Same version/regression

Mark duplicate=true ONLY if root cause is very likely identical.

Return JSON:
[{{"number": int, "duplicate": true/false}}]
"""
    # ---------------------------
    # Build Prompt
    # ---------------------------
    for c in candidates:
        prompt += f"\nIssue {c['number']}: {c['title']} - {c['body'][:150]}"


    # ---------------------------
    # Token Count
    # ---------------------------
    try:
        token_data = llm.count_tokens("gemini", MODEL_NAME, prompt)
        token_count = token_data["total_tokens"]
    except Exception:
        token_count = 0


    # ---------------------------
    # LLM Call
    # ---------------------------
    try:
        result_obj = llm.generate("gemini", MODEL_NAME, prompt)
        txt = result_obj["text"]

        match = re.search(r"\[.*?\]", txt, re.S)
        data = json.loads(match.group(0)) if match else []

        dup_map = {d["number"]: d["duplicate"] for d in data}

        for c in candidates:
            c["llm_duplicate"] = dup_map.get(c["number"], False)

    except Exception:
        pass


    return candidates, token_count


# ==========================
# SEMANTIC MATCH
# ==========================
def overlap(a, b):
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)


def build_embedding_text(issue):
    body = issue.get("body") or ""

    errors = "\n".join(re.findall(r"(?:Error|Exception).*", body)[:3])
    stack = "\n".join(re.findall(r".*\.(ts|js|py):\d+.*", body)[:5])

    return f"""
Title: {issue['title']}

Errors:
{errors}

Stacktrace:
{stack}

Summary:
{clean_issue_text(body)[:300]}
"""



def semantic_issue_matcher(
    query_title,
    query_body,
    candidates,
    exclude_issue_number,
    top_k=10,
    min_score=0.55
):
    query_signals = extract_signals(query_title + " " + query_body)
    query_text = clean_issue_text(query_title + " " + query_body)


    texts = [build_embedding_text({"title": query_title, "body": query_body})] + [
        build_embedding_text(c) for c in candidates
    ]

    vectors = embed_texts_batch(texts)
    q_vec = vectors[0]

    results, duplicates = [], []

    has_stack = detect_stacktrace(query_body)

    for issue, vec in zip(candidates, vectors[1:]):
        issue_signals = extract_signals(issue["title"] + issue.get("body", ""))
        if issue["number"] == exclude_issue_number:
            continue

        sem = cosine_similarity(q_vec, vec)
        lex = jaccard(query_text, issue["title"] + issue.get("body", ""))

        func_score = overlap(query_signals["functions"], issue_signals["functions"])
        file_score = overlap(query_signals["files"], issue_signals["files"])
        error_score = overlap(query_signals["errors"], issue_signals["errors"])

        structural = (0.5 * func_score) + (0.3 * file_score) + (0.2 * error_score)

        score = (0.6 * sem) + (0.2 * lex) + (0.2 * structural)

        if has_stack:
            score += 0.05

        record = {
            **issue,
            "score": round(score, 4),
            "semantic": round(sem, 4),
            "lexical": round(lex, 4)
        }

        if sem > 0.85 and structural > 0.5:
            duplicates.append(record)
        elif score >= min_score:
            results.append(record)

    results.sort(key=lambda x: x["score"], reverse=True)

    top_candidates = results[:MAX_LLM_CHECK]
    top_candidates, token_count = llm_rerank(query_title, query_body, top_candidates)

    duplicates += [r for r in top_candidates if r.get("llm_duplicate")]

    return {
        "duplicates": duplicates,
        "similar_fixes": results[:top_k]
    }, token_count


def github_get_issue_details(
    installation_id: int,
    owner: str,
    repo: str,
    issue_number: int
):
    token = get_installation_token(installation_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"GitHub API error: {response.status_code} - {response.text}"
        )

    data = response.json()

    return {
        "id": data["id"],
        "number": data["number"],
        "title": data["title"],
        "body": data.get("body") or "",
        "state": data["state"],
        "user": data["user"]["login"],
        "labels": [label["name"] for label in data.get("labels", [])],
        "comments_count": data["comments"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "closed_at": data.get("closed_at"),
        "html_url": data["html_url"]
    }
    



def github_get_issue_comments(
    installation_id: int,
    owner: str,
    repo: str,
    issue_number: int,
    per_page: int = 30,
    max_comments: int = 100
) -> List[Dict]:
    """
    Fetch comments for a GitHub issue.

    Supports pagination and limits total comments fetched.
    """

    token = get_installation_token(installation_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"

    comments = []
    page = 1

    while len(comments) < max_comments:

        params = {
            "per_page": per_page,
            "page": page
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(
                f"GitHub API error: {response.status_code} - {response.text}"
            )

        data = response.json()

        # No more comments
        if not data:
            break

        for c in data:
            comments.append({
                "id": c["id"],
                "user": c["user"]["login"],
                "body": c.get("body") or "",
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
                "html_url": c["html_url"]
            })

            if len(comments) >= max_comments:
                break

        page += 1

    return comments


# ==========================
# MAIN PIPELINE
# ==========================

def maintania_find_similar_fixes(
    installation_id,
    owner,
    repo,
    issue_number,
    title,
    body,
    top_k=10
):

    keywords = extract_keywords(title, body)
    llm_keywords, tk1 = llm_extract_keywords(title, body)

    keywords = list(dict.fromkeys(keywords + llm_keywords))[:10]

    signals = build_signals_from_keywords(keywords)  # ✅ NEW

    candidates = github_recent_keyword_search(
        installation_id,
        owner,
        repo,
        signals,   # ✅ pass signals instead of keywords
        exclude_issue_number=issue_number
    )

    if not candidates:
        return {
            "issue_number": issue_number,
            "duplicates": [],
            "similar_fixes": []
        }

    results, tk2 = semantic_issue_matcher(
        title, body, candidates, issue_number, top_k
    )

    return {
        "issue_number": issue_number,
        "llm_keywords": llm_keywords,
        "token_count_keywords": tk1,
        "token_count_rank": tk2,
        **results
    }