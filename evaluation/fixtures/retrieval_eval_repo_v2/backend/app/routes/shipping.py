from fastapi import APIRouter
from backend.app.services import shipping_service

router = APIRouter()

@router.get("/api/shipping/{order_id}")
def shipment_status(order_id: str):
    return shipping_service.track_order(order_id)
