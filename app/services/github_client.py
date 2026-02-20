import time
import jwt
import requests
from app.core.config import settings


def generate_jwt():
    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "iss": settings.GITHUB_APP_ID,
    }

    return jwt.encode(
        payload,
        settings.GITHUB_APP_PRIVATE_KEY,
        algorithm="RS256"
    )


def get_installation_token(installation_id: str):
    jwt_token = generate_jwt()

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
    }

    res = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers,
    )

    data = res.json()

    if "token" not in data:
        print("GitHub token error:", data)
        raise Exception(f"Failed to get installation token: {data}")

    return data["token"]

def get_installation_repos(installation_id):

    token = get_installation_token(installation_id)
    url = "https://api.github.com/installation/repositories"

    res = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
    )

    return res.json()


def comment_on_issue(installation_id, owner, repo, issue_number, comment_body):

    token = get_installation_token(installation_id)

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"

    res = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        },
        json={
            "body": comment_body
        }
    )

    return res.json()
