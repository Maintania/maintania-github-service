import time
import re
import numpy as np
import requests
from typing import List, Dict
from itertools import combinations
from google import genai
from app.services.github_client import get_installation_token
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import os

login(token=os.getenv("HF_TOKEN"))


# ==========================
# Safe Batched Embedding
# ==========================

model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code")

def embed_texts_batch(texts: list[str]):
    print('starts embeddings')
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()


# ==========================
# Similarity Metrics
# ==========================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def jaccard(a: str, b: str):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# ==========================
# Keyword Extraction
# ==========================

def extract_keywords(title: str, body: str, k=8):
    text = f"{title} {body}".lower()

    words = re.findall(r"[a-z]{3,}", text)

    stop = {
        "this","that","with","from","have","your","will","when",
        "what","which","there","about","where","would","could",
        "their","they","them","then","than","into","onto","using"
    }

    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1

    return sorted(freq, key=freq.get, reverse=True)[:k]


# ==========================
# Keyword Phrase Generator
# ==========================

def keyword_phrases(keywords, max_phrases=6):
    phrases = []

    for k in (2, 3):
        for combo in combinations(keywords, k):
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
    per_query: int = 15
):
    token = get_installation_token(installation_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    url = "https://api.github.com/search/issues"
    results = {}

    phrases = keyword_phrases(keywords)
    print(len(phrases))

    for phrase in phrases:
        q = f'repo:{owner}/{repo} {phrase}'

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
        print("Search:", q, "=>", res.status_code)

        if res.status_code != 200:
            continue

        for item in res.json().get("items", []):

            # 🚨 Exclude self issue
            if item["number"] == exclude_issue_number:
                continue

            results[item["id"]] = {
                "number": item["number"],
                "title": item["title"],
                "body": item.get("body") or "",
                "html_url": item["html_url"],
                "created_at": item["created_at"]
            }

    return list(results.values())


# ==========================
# Semantic Ranking + Duplicate Detection
# ==========================

def semantic_issue_matcher(
    query_title: str,
    query_body: str,
    candidates: List[Dict],
    exclude_issue_number: int,
    top_k: int = 10,
    min_score: float = 0.55
):
    print('started scoring semantic_issue_matcher')
    query_text = f"{query_title}\n\n{query_body}".strip()

    candidate_texts = [
        f"{c['title']}\n\n{c.get('body','')}"[:4000]
        for c in candidates
    ]

    vectors = embed_texts_batch([query_text] + candidate_texts)
    print('embedding done')
    q_vec = vectors[0]
    issue_vecs = vectors[1:]

    duplicates = []
    results = []

    for issue, vec in zip(candidates, issue_vecs):
        print(issue["number"])
        if issue["number"] == exclude_issue_number:
            continue

        sem = cosine_similarity(q_vec, vec)
        lex = jaccard(query_text, issue["title"] + " " + issue.get("body", ""))

        score = 0.75 * sem + 0.25 * lex

        record = {
            **issue,
            "score": round(score, 4),
            "semantic": round(sem, 4),
            "lexical": round(lex, 4)
        }

        # 🚨 Exact duplicate detection
        if sem > 0.97 and lex > 0.75:
            duplicates.append(record)
            continue

        if score >= min_score:
            results.append(record)

    results.sort(key=lambda x: x["score"], reverse=True)
    duplicates.sort(key=lambda x: x["semantic"], reverse=True)

    return {
        "duplicates": duplicates,
        "similar_fixes": results[:top_k]
    }


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

    candidates = github_recent_keyword_search(
        installation_id,
        owner,
        repo,
        keywords,
        exclude_issue_number=issue_number
    )
    print(len(candidates))
    if not candidates:
        return {
            "issue_number": issue_number,
            "duplicates": [],
            "similar_fixes": []
        }

    results = semantic_issue_matcher(
        query_title=title,
        query_body=body,
        candidates=candidates,
        exclude_issue_number=issue_number,
        top_k=top_k
    )

    return {
        "issue_number": issue_number,
        **results
    }