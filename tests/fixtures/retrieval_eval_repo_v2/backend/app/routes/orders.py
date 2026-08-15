from fastapi import APIRouter
from backend.app.services import order_service

router = APIRouter()

@router.post("/api/orders")
def create_order(payload: dict):
    return order_service.place_order(payload)

@router.get("/api/orders/{order_id}")
def get_order(order_id: str):
    return order_service.load_order(order_id)

@router.delete("/api/orders/{order_id}")
def cancel_order(order_id: str):
    return order_service.cancel_if_allowed(order_id)
