from pydantic import BaseModel

class UserRead(BaseModel):
    id: int
    username: str | None
    name: str | None
    avatar_url: str | None

    class Config:
        from_attributes = True
