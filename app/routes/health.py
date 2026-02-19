# THIS ROUTE IS FOR CHECKING HEALTH OF THE SYSTEM DO NOT TOUCH !! FIRE HU MA
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def health():
    return {"status": "github automation running"}
