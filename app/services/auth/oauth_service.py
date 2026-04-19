import requests
from app.core.config import settings

class OAuthService:

    @staticmethod
    def exchange_code(code: str):
        res = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code
            },
            timeout=5
        )

        return res.json()["access_token"]

    @staticmethod
    def get_user(token: str):
        res = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )

        return res.json()