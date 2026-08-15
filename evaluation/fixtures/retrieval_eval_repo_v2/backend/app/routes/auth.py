from fastapi import APIRouter
from backend.app.services import auth_service

router = APIRouter()

@router.post("/api/auth/login")
def login(credentials: dict):
    return auth_service.authenticate_user(credentials)

@router.post("/api/auth/refresh")
def refresh_session(payload: dict):
    return auth_service.rotate_session(payload["refresh_token"])
