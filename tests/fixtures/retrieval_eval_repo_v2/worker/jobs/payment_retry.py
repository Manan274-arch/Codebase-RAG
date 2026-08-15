from backend.app.services.payment_gateway import process_charge

def retry_deferred_charge(payload: dict, attempt: int) -> dict:
    if attempt > 3:
        raise RuntimeError("payment retry exhausted")
    return process_charge(payload)
