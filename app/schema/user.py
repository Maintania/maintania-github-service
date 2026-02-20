from pydantic import BaseModel

class UserSchema(BaseModel):
    github_id: str
    username: str | None = None
    name: str | None = None
    email: str | None = None
    image: str | None = None
