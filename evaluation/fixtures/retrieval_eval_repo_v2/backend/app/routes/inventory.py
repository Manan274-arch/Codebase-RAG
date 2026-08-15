from fastapi import APIRouter
from backend.app.services import inventory_service

router = APIRouter()

@router.get("/api/inventory/{sku}")
def inventory_status(sku: str):
    return inventory_service.available_units(sku)

@router.patch("/api/inventory/{sku}/reserve")
def reserve_inventory(sku: str, payload: dict):
    return inventory_service.reserve_stock(sku, payload["quantity"])
