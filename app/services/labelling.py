import json
import requests
import os
from typing import List

# ================== CONFIG ==================

# It's safer to use an environment variable: export GEMINI_API_KEY='your_key'
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY")

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
        "backend": ["backend"],
        "api": ["api"],
        "database": ["database"],
        "auth": ["auth"],
        "ai": ["ai"],
        "infra": ["infra"],
        "testing": ["testing"]
    },
    "priority": {
        "low": ["priority:low"],
        "medium": ["priority:medium"],
        "high": ["priority:high"],
        "critical": ["priority:critical"]
    }
}

# ================== FUNCTION ==================

def classify_issue(title: str, body: str) -> dict:
    """
    Optimized classification using Gemini 2.5 Flash-Lite (Lowest Cost)
    and fixed JSON pathing.
    """
    # Using the 'latest' alias for Flash-Lite ensures you get the most cost-effective model
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    params = {"key": LLM_API_KEY}
    
    response_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(LABEL_MAP["type"].keys())},
            "area": {"type": "array", "items": {"type": "string", "enum": list(LABEL_MAP["area"].keys())}},
            "priority": {"type": "string", "enum": list(LABEL_MAP["priority"].keys())},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"}
        },
        "required": ["type", "area", "priority", "confidence", "reasoning"]
    }

    # Truncate body to 2000 chars to save on input tokens (Cost Optimization)
    truncated_body = (body[:2000] + '...') if len(body) > 2000 else body
    prompt = f"Classify this GitHub issue:\nTitle: {title}\nBody: {truncated_body}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 300, # Prevents long-winded answers that cost more
            "response_mime_type": "application/json",
            "response_schema": response_schema
        }
    }

    response = requests.post(url, params=params, json=payload, timeout=30)
    
    if response.status_code != 200:
        print(f"API Error: {response.text}")
        response.raise_for_status()

    res_json = response.json()

    # ================= FIXED PATHING HERE =================
    # We must use because 'candidates' and 'parts' are lists
    try:
        print(res_json) 
        raw_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
        llm_data = json.loads(raw_content)
    except (KeyError, IndexError) as e:
        print(f"Failed to parse LLM response: {e}")
        return {"error": "Invalid API response structure"}
    # ======================================================

    # Mapping logic
    github_labels = set()
    github_labels.update(LABEL_MAP["type"].get(llm_data.get("type"), []))
    github_labels.update(LABEL_MAP["priority"].get(llm_data.get("priority"), []))
    
    for area in llm_data.get("area", []):
        github_labels.update(LABEL_MAP["area"].get(area, []))

    llm_data["github_labels"] = list(github_labels)
    
    return llm_data


# # ================== EXAMPLE USAGE ==================

# if __name__ == "__main__":
#     example_title = "App crashes when clicking 'Save' on the profile page"
#     example_body = "Steps to reproduce: 1. Go to profile. 2. Edit name. 3. Click Save. The app immediately closes without an error message."

#     try:
#         result = classify_issue(example_title, example_body)
#         print(json.dumps(result, indent=2))
#     except Exception as e:
#         print(f"Error: {e}")