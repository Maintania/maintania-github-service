import json
import requests
import os
import re
from typing import List, Dict, Any

# ================== CONFIG ==================

LLM_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

LABEL_MAP = {
    "type": {
        "bug": ["bug"],
        "feature": ["feature-request"],
        "enhancement": ["enhancement"],
        "docs": ["documentation"],
        "performance": ["performance"],
        "security": ["security"],
        "question": ["question"]
    },
    "area": {
        "ui": ["ui"],
        "frontend": ["frontend"],
        "backend": ["backend"],
        "api": ["api"],
        "cli": ["cli"],
        "chat": ["chat"],
        "extensions": ["extensions"],
        "terminal": ["terminal"],
        "editor": ["editor"],
        "database": ["database"],
        "auth": ["auth"],
        "storage": ["storage"],
        "network": ["network"],
        "build": ["build"],
        "infra": ["infra"],
        "testing": ["testing"],
        "docs": ["documentation"]
    },
    "priority": {
        "low": ["priority:low"],
        "medium": ["priority:medium"],
        "high": ["priority:high"],
        "critical": ["priority:critical"]
    }
}

# ================== TEXT CLEANING ==================

def clean_issue_text(text: str) -> str:
    """
    Removes GitHub issue template noise.
    """
    if not text:
        return ""

    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    patterns = [
        r"Does this issue occur when.*",
        r"VS Code Version:.*",
        r"OS Version:.*",
        r"Steps to Reproduce:.*"
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.I)

    return text.strip()


# ================== HEURISTIC AREA DETECTION ==================

def heuristic_labels(title: str, body: str) -> List[str]:
    text = f"{title} {body}".lower()

    areas = set()

    if "terminal" in text:
        areas.add("terminal")

    if "extension" in text:
        areas.add("extensions")

    if "chat" in text:
        areas.add("chat")

    if "api" in text:
        areas.add("api")

    if "database" in text or "sql" in text:
        areas.add("database")

    if "auth" in text or "login" in text:
        areas.add("auth")

    if "editor" in text:
        areas.add("editor")

    if "build" in text or "compile" in text:
        areas.add("build")

    return list(areas)


# ================== PRIORITY HEURISTICS ==================

def infer_priority(text: str) -> str:
    text = text.lower()

    if "crash" in text or "data loss" in text:
        return "critical"

    if "error" in text or "fails" in text:
        return "high"

    if "slow" in text or "performance" in text:
        return "medium"

    return "low"


# ================== LLM CLASSIFICATION ==================

def llm_classify(title: str, body: str) -> Dict[str, Any]:

    response_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(LABEL_MAP["type"].keys())},
            "area": {
                "type": "array",
                "items": {"type": "string", "enum": list(LABEL_MAP["area"].keys())}
            },
            "priority": {"type": "string", "enum": list(LABEL_MAP["priority"].keys())},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"}
        },
        "required": ["type", "area", "priority", "confidence", "reasoning"]
    }

    truncated_body = (body[:2000] + "...") if len(body) > 2000 else body

    prompt = f"""
You are a GitHub issue triage assistant.

Classify the issue into:
1) type
2) area
3) priority

Rules:
- bug = crashes, errors, incorrect behavior
- feature = request for new functionality
- enhancement = improvement to existing feature
- docs = documentation problems
- performance = slow operations
- security = vulnerabilities

Return JSON only.

Issue Title:
{title}

Issue Body:
{truncated_body}
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 300,
            "response_mime_type": "application/json",
            "response_schema": response_schema
        }
    }

    response = requests.post(
        GEMINI_URL,
        params={"key": LLM_API_KEY},
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    res_json = response.json()

    raw_content = res_json["candidates"][0]["content"]["parts"][0]["text"]

    return json.loads(raw_content)


# ================== MAIN PIPELINE ==================

def classify_issue(title: str, body: str) -> Dict[str, Any]:

    body = clean_issue_text(body)

    heuristic_area = heuristic_labels(title, body)

    llm_data = llm_classify(title, body)

    # merge heuristic areas
    areas = set(llm_data["area"])

    for a in heuristic_area:
        areas.add(a)

    llm_data["area"] = list(areas)

    # priority guard
    if llm_data["confidence"] < 0.6:
        llm_data["priority"] = infer_priority(title + body)

    # map GitHub labels
    github_labels = set()

    github_labels.update(LABEL_MAP["type"].get(llm_data["type"], []))
    github_labels.update(LABEL_MAP["priority"].get(llm_data["priority"], []))

    for area in llm_data["area"]:
        github_labels.update(LABEL_MAP["area"].get(area, []))

    llm_data["github_labels"] = list(github_labels)

    return llm_data


# # ================== TEST ==================

# if __name__ == "__main__":

#     issue = {
#         "title": "Error serializing chat session for storage",
#         "body": """
# Does this issue occur when all extensions are disabled?: Yes/No\n\nVS Code Version:\nOS Version:\nSteps to Reproduce:\n\nError serializing chat session for storage. The session will be lost if the window is closed. Please report this issue to the VS Code team:\n\nError: error diffing at .requests[9].response: Unreachable\nat Br (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:386:162)\nat Object.equals (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:2708)\nat Zte._diffArray (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:1491)\nat Zte._diff (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:572)\nat Zte._diffObject (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:967)\nat Zte._diffArray (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:1285)\nat Zte._diff (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:572)\nat Zte._diffObject (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:967)\nat Zte._diff (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:618)\nat Zte.write (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2909:28)\nat eie.writeSession (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:11578)\nat vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:8538\nat Array.map ()\nat vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:8526\nat vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:406:73746\nat async eie.storeSessions (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:8683)\nat async Object.willDisposeModel (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2910:22036)\nat async COe.doDestroyReferencedObject (vscode-file://vscode-app/c:/Users/maraf/AppData/Local/Programs/Microsoft%20VS%20Code/ce099c1ed2/resources/app/out/vs/workbench/workbench.desktop.main.js:2904:84009)
# """
#     }

#     result = classify_issue(issue["title"], issue["body"])

#     print(json.dumps(result, indent=2))