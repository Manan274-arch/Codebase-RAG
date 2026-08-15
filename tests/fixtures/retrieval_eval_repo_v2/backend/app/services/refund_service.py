from backend.app.repositories import refund_repository, payment_repository

def open_refund(payload: dict) -> dict:
    return refund_repository.insert_request(payload)

def approve_refund(refund_id: str, admin_id: str) -> dict:
    refund = refund_repository.mark_approved(refund_id, admin_id)
    payment_repository.return_funds(refund["payment_id"], refund["amount"])
    return refund
