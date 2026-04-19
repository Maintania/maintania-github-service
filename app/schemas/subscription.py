from pydantic import BaseModel

class SubscriptionRead(BaseModel):
    id: int
    status: str
    plan_id: int

    class Config:
        from_attributes = True