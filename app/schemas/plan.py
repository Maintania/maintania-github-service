from pydantic import BaseModel

class PlanRead(BaseModel):
    id: int
    name: str
    price: float
    interval: str
    max_repos: int
    max_requests: int

    class Config:
        from_attributes = True