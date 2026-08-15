from backend.app.services import discount_service, order_service, payment_gateway

def complete_checkout(payload: dict) -> dict:
    total = discount_service.final_total(payload["items"], payload.get("coupon"))
    authorization = payment_gateway.process_charge({"amount": total, "token": payload["payment_token"]})
    return order_service.place_order({**payload, "total": total, "authorization": authorization["id"]})
