def queue_order_confirmation(order: dict) -> dict:
    return {"job": "order_confirmation", "order_id": order["id"]}

def queue_shipping_update(shipment: dict) -> dict:
    return {"job": "shipping_update", "shipment_id": shipment["id"]}
