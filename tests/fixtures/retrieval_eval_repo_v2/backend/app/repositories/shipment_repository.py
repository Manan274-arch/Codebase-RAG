SHIPMENTS: dict[str, dict] = {}

def create_shipment(order_id: str, address: dict) -> dict:
    shipment = {"id": f"ship_{len(SHIPMENTS)+1}", "order_id": order_id, "address": address, "status": "queued"}
    SHIPMENTS[shipment["id"]] = shipment
    return shipment

def find_by_order(order_id: str) -> dict:
    return next(item for item in SHIPMENTS.values() if item["order_id"] == order_id)
