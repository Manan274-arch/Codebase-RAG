from fastapi import APIRouter
from backend.app.services import payment_gateway

router = APIRouter()

@router.post("/api/payments/authorize")
def authorize_payment(payload: dict):
    return payment_gateway.process_charge(payload)

@router.post("/api/payments/webhook")
def payment_webhook(event: dict):
    return payment_gateway.accept_provider_event(event)
