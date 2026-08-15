ORDERS: dict[str, dict] = {}

def insert_order(payload: dict) -> dict:
    order_id = f"ord_{len(ORDERS) + 1}"
    ORDERS[order_id] = {"id": order_id, "status": "confirmed", **payload}
    return ORDERS[order_id]

def get_order(order_id: str) -> dict:
    return ORDERS[order_id]

def set_status(order_id: str, status: str) -> dict:
    ORDERS[order_id]["status"] = status
    return ORDERS[order_id]
