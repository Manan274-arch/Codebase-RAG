from backend.app.repositories import payment_repository

TRANSIENT_CODES = {"timeout", "rate_limited"}

def process_charge(request: dict) -> dict:
    response = _provider_authorize(request)
    if response["status"] in TRANSIENT_CODES:
        payment_repository.mark_for_retry(request)
        raise RuntimeError("payment temporarily unavailable")
    if response["status"] != "approved":
        raise ValueError("payment authorization failed")
    return payment_repository.record_authorization(response)

def accept_provider_event(event: dict) -> dict:
    if payment_repository.webhook_seen(event["id"]):
        return {"duplicate": True}
    payment_repository.record_webhook(event)
    return {"accepted": True}

def _provider_authorize(request: dict) -> dict:
    return {"id": "auth_generated", "status": "approved", **request}
