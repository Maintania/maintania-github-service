import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
    GITHUB_APP_PRIVATE_KEY = os.getenv("GITHUB_APP_PRIVATE_KEY")
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI")
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
    MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL")
    DATABASE_URL = os.getenv("DATABASE_URL")
    FRONTEND_URL = os.getenv("FRONTEND_URL")

settings = Settings()
