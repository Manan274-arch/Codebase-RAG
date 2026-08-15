from backend.app.services.shipping_service import schedule_dispatch

def dispatch_confirmed_order(order: dict) -> dict:
    return schedule_dispatch(order["id"], order["shipping_address"])
