def build_order(overrides: dict | None = None) -> dict:
    return {"id": "test-order", "status": "pending", **(overrides or {})}
