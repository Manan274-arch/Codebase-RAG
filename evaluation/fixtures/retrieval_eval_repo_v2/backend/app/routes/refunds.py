from fastapi import APIRouter
from backend.app.services import refund_service

router = APIRouter()

@router.post("/api/refunds")
def request_refund(payload: dict):
    return refund_service.open_refund(payload)
