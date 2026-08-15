from fastapi import APIRouter
from backend.app.services import refund_service

router = APIRouter()

@router.post("/api/admin/refunds/{refund_id}/approve")
def approve_refund(refund_id: str, admin_id: str):
    return refund_service.approve_refund(refund_id, admin_id)
