from fastapi import APIRouter
from backend.app.services import profile_service

router = APIRouter()

@router.get("/api/profile")
def current_profile(user_id: str):
    return profile_service.load_profile(user_id)

@router.patch("/api/profile")
def update_profile(user_id: str, changes: dict):
    return profile_service.update_profile(user_id, changes)
