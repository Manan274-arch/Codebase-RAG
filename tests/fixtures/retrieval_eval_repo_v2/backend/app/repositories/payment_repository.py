AUTHORIZATIONS: dict[str, dict] = {}
WEBHOOKS: set[str] = set()

def record_authorization(response: dict) -> dict:
    AUTHORIZATIONS[response["id"]] = response
    return response

def mark_for_retry(request: dict) -> None:
    AUTHORIZATIONS[request["token"]] = {**request, "status": "retry"}

def webhook_seen(event_id: str) -> bool:
    return event_id in WEBHOOKS

def record_webhook(event: dict) -> None:
    WEBHOOKS.add(event["id"])

def return_funds(payment_id: str, amount: float) -> dict:
    return {"payment_id": payment_id, "refunded": amount}
