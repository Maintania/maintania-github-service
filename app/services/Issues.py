import re
import numpy as np
import requests
import os
import json
from typing import List, Dict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import login

from app.services.github_client import get_installation_token
from app.services.repo_cloner import embed

from google import genai


login(token=os.getenv("HF_TOKEN"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

MAX_LLM_CHECK = 5


client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


# ==========================
# Safe Batched Embedding
# ==========================

def embed_texts_batch(texts: list[str]):
    print("starts embeddings for", len(texts), "texts")
    embeddings = embed(texts)
    return embeddings.tolist()


# ==========================
# Similarity Metrics
# ==========================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def jaccard(a: str, b: str):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# ==========================
# Stacktrace Detection
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

    for p in patterns:
        if re.search(p, text.lower()):
            return True

    return False


# ==========================
# Text Cleaning
# ==========================

def clean_issue_text(text: str):

    if not text:
        return ""

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    return text


# ==========================
# Keyword Extraction
# ==========================

def extract_keywords(title: str, body: str, k=8):

    body = clean_issue_text(body)

    text = f"{title} {body}".lower()

    words = re.findall(r"[a-z]{3,}", text)

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
# Gemini Keyword Extraction (optional)
# ==========================
def clean_llm_json(text: str):

    text = text.strip()

    # remove markdown code blocks
    text = text.replace("```json", "")
    text = text.replace("```", "")

    text = text.strip()

    return json.loads(text)

def llm_extract_keywords(title: str, body: str):

    if not GEMINI_API_KEY:
        return []

    prompt = f"""
        Extract 5 important search keywords from this GitHub issue.

        Title:
        {title}

        Body:
        {body[:800]}

        Strictly follow the format. Return JSON list.
        ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    """

    # =====================
    # TOKEN COUNT
    # =====================
    token_count = 0
    try:
        token_info = client.models.count_tokens(
            model=MODEL_NAME,
            contents=prompt
        )

        print("Gemini keyword extraction tokens:", token_info.total_tokens)
        token_count = token_info.total_tokens
    except Exception as e:
        print("Token count failed:", e)

    payload = {
        "contents":[{"parts":[{"text":prompt}]}],
        "generationConfig":{
            "temperature":0,
            "maxOutputTokens":60
        }
    }

    try:

        r = requests.post(
            GEMINI_URL,
            params={"key":GEMINI_API_KEY},
            json=payload,
            timeout=20
        )
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]

        print("Raw Gemini response:", text)

        keywords = clean_llm_json(text)

        return keywords, token_count

    except:
        return [],token_count


# ==========================
# Keyword Phrase Generator
# ==========================

def keyword_phrases(keywords, max_phrases=6):

    phrases = []

    for k in (2,3):
        for combo in combinations(keywords,k):
            phrases.append(" ".join(combo))

    return phrases[:max_phrases]


# ==========================
# GitHub Issue Search
# ==========================

def github_recent_keyword_search(
    installation_id: int,
    owner: str,
    repo: str,
    keywords: List[str],
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

    phrases = keyword_phrases(keywords)

    with ThreadPoolExecutor(max_workers=6) as executor:

        futures = []

        for phrase in phrases:

            futures.append(
                executor.submit(
                    search_phrase,
                    url,
                    headers,
                    owner,
                    repo,
                    phrase,
                    per_query
                )
            )

        for future in as_completed(futures):

            items = future.result()

            for item in items:

                if item["number"] == exclude_issue_number:
                    continue

                results[item["id"]] = {
                    "number": item["number"],
                    "title": item["title"],
                    "body": item.get("body") or "",
                    "html_url": item["html_url"],
                    "created_at": item["created_at"]
                }

    candidates = list(results.values())

    unique_candidates = {}
    for c in candidates:
        unique_candidates[c["number"]] = c

    return list(unique_candidates.values())


def search_phrase(url, headers, owner, repo, phrase, per_query):

    q = f"repo:{owner}/{repo} {phrase} in:title,body"

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


# ==========================
# Gemini Final Reranker
# ==========================
def llm_rerank(query_title, query_body, candidates):

    token_count = 0
    if not GEMINI_API_KEY or not candidates:
        return candidates,token_count
    prompt = f"""
        A GitHub issue is given.

        Title:
        {query_title}

        Body:
        {query_body[:800]}

        Below are candidate similar issues.

        Rank them by similarity and detect duplicates.

        Return JSON:
        [
        {{"number": issue_number, "duplicate": true/false}}
        ]
    """

    for c in candidates:
        prompt += f"\nIssue {c['number']}: {c['title']}"

    # =====================
    # TOKEN COUNT
    # =====================

    try:
        token_info = client.models.count_tokens(
            model=MODEL_NAME,
            contents=prompt
        )

        print("Gemini rerank tokens:", token_info.total_tokens)
        token_count = token_info.total_tokens
    except Exception as e:
        print("Token count failed:", e)

    payload = {
        "contents":[{"parts":[{"text":prompt}]}],
        "generationConfig":{
            "temperature":0,
            "maxOutputTokens":200
        }
    }

    try:

        r = requests.post(
            GEMINI_URL,
            params={"key":GEMINI_API_KEY},
            json=payload,
            timeout=30
        )

        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]

        data = json.loads(txt)

        dup = {d["number"]: d["duplicate"] for d in data}

        for c in candidates:
            c["llm_duplicate"] = dup.get(c["number"], False)

    except:
        pass

    return candidates,token_count


# ==========================
# Semantic Ranking
# ==========================

def semantic_issue_matcher(
    query_title: str,
    query_body: str,
    candidates: List[Dict],
    exclude_issue_number: int,
    top_k: int = 10,
    min_score: float = 0.55
):

    query_text = f"{query_title}\n\n{query_body}".strip()

    candidate_texts = [
        f"{c['title']}\n\n{(c.get('body') or '')[:800]}"
        for c in candidates
    ]

    vectors = embed_texts_batch([query_text] + candidate_texts)

    q_vec = vectors[0]
    issue_vecs = vectors[1:]

    results = []
    duplicates = []

    for issue, vec in zip(candidates, issue_vecs):

        if issue["number"] == exclude_issue_number:
            continue

        sem = cosine_similarity(q_vec, vec)

        lex = jaccard(
            query_text,
            issue["title"] + " " + issue.get("body", "")
        )

        score = 0.75 * sem + 0.25 * lex

        record = {
            **issue,
            "score": round(score,4),
            "semantic": round(sem,4),
            "lexical": round(lex,4)
        }

        if sem > 0.96 and lex > 0.7:
            duplicates.append(record)
            continue

        if score >= min_score:
            results.append(record)

    results.sort(key=lambda x:x["score"], reverse=True)

    # final LLM rerank
    top_candidates = results[:MAX_LLM_CHECK]

    top_candidates,token_count = llm_rerank(query_title, query_body, top_candidates)

    for r in top_candidates:
        if r.get("llm_duplicate"):
            duplicates.append(r)

    return {
        "duplicates": duplicates,
        "similar_fixes": results[:top_k]
    },token_count


# ==========================
# Final Maintania Pipeline
# ==========================

def maintania_find_similar_fixes(
    installation_id: int,
    owner: str,
    repo: str,
    issue_number: int,
    title: str,
    body: str,
    top_k: int = 10
):
    keywords = extract_keywords(title, body)

    llm_keywords,token_count_key = llm_extract_keywords(title, body)

    keywords = list(dict.fromkeys(keywords + llm_keywords))[:10]

    candidates = github_recent_keyword_search(
        installation_id,
        owner,
        repo,
        keywords,
        exclude_issue_number=issue_number
    )

    if not candidates:
        return {
            "issue_number": issue_number,
            "duplicates": [],
            "similar_fixes": []
        }

    results,token_count_rank = semantic_issue_matcher(
        query_title=title,
        query_body=body,
        candidates=candidates,
        exclude_issue_number=issue_number,
        top_k=top_k
    )

    
    return {
        "issue_number": issue_number,
        "llm_keywords": llm_keywords,
        "token_count_keywords": token_count_key,
        "token_count_rank": token_count_rank,
        **results
    }
