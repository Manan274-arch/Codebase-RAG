def fake_provider_authorize(payload: dict) -> dict:
    return {"id": "mock", "status": "approved", **payload}
