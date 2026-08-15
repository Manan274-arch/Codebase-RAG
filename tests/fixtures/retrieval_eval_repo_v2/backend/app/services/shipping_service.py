from backend.app.repositories import shipment_repository

def track_order(order_id: str) -> dict:
    return shipment_repository.find_by_order(order_id)

def schedule_dispatch(order_id: str, address: dict) -> dict:
    return shipment_repository.create_shipment(order_id, address)
