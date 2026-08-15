REFUNDS: dict[str, dict] = {}

def insert_request(payload: dict) -> dict:
    refund_id = f"ref_{len(REFUNDS)+1}"
    REFUNDS[refund_id] = {"id": refund_id, "status": "requested", **payload}
    return REFUNDS[refund_id]

def mark_approved(refund_id: str, admin_id: str) -> dict:
    REFUNDS[refund_id].update(status="approved", approved_by=admin_id)
    return REFUNDS[refund_id]
