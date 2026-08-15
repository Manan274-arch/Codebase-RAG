from backend.app.repositories import inventory_repository

def available_units(sku: str) -> int:
    return inventory_repository.quantity_for(sku)

def reserve_stock(sku: str, quantity: int) -> dict:
    return inventory_repository.decrement_available(sku, quantity)

def reserve_items(items: list[dict]) -> None:
    for item in items:
        reserve_stock(item["sku"], item["quantity"])

def release_items(items: list[dict]) -> None:
    for item in items:
        inventory_repository.increment_available(item["sku"], item["quantity"])
