import time
import jwt
import requests
from app.core.config import settings
import os 



def generate_jwt():
    payload = {
        "iat": int(time.time())- 60,
        "exp": int(time.time()) + 540,
        "iss": settings.GITHUB_APP_ID,
    }

    return jwt.encode(
        payload,
        settings.GITHUB_APP_PRIVATE_KEY,
        algorithm="RS256"
    )


def get_installation_token(installation_id: str):
    jwt_token = generate_jwt()

    res = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        },
    )

    data = res.json()

    print("GitHub token response:", data)

    if "token" not in data:
        raise Exception(f"GitHub token error: {data}")

    return data["token"]
    # return os.getenv("N_Access_Token")



def get_installation_repos(installation_id: str):

    token = get_installation_token(installation_id)

    url = "https://api.github.com/installation/repositories"
    repos = []

    page = 1

    while True:
        res = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json"
            },
            params={"per_page": 100, "page": page}
        )

        if res.status_code != 200:
            raise Exception(f"GitHub API error {res.status_code}: {res.text}")

        data = res.json()
        batch = data.get("repositories", [])

        repos.extend(batch)

        if len(batch) < 100:
            break

        page += 1
    print("GitHub response:", res.status_code, res.text)
    return {"repositories": repos}


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