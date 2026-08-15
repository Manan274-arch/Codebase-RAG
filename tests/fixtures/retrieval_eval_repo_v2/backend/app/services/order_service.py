from backend.app.repositories import order_repository
from backend.app.services import inventory_service, notification_service

def place_order(payload: dict) -> dict:
    inventory_service.reserve_items(payload["items"])
    order = order_repository.insert_order(payload)
    notification_service.queue_order_confirmation(order)
    return order

def load_order(order_id: str) -> dict:
    return order_repository.get_order(order_id)

def cancel_if_allowed(order_id: str) -> dict:
    order = order_repository.get_order(order_id)
    if order["status"] not in {"pending", "confirmed"}:
        raise ValueError("order can no longer be cancelled")
    inventory_service.release_items(order["items"])
    return order_repository.set_status(order_id, "cancelled")
