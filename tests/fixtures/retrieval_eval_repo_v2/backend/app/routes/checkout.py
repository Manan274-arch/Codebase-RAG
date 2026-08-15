from fastapi import APIRouter
from backend.app.services import checkout_service

router = APIRouter()

@router.post("/api/checkout")
def submit_checkout(payload: dict):
    return checkout_service.complete_checkout(payload)
