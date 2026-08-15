STOCK: dict[str, int] = {}

def quantity_for(sku: str) -> int:
    return STOCK.get(sku, 0)

def decrement_available(sku: str, quantity: int) -> dict:
    if quantity_for(sku) < quantity:
        raise ValueError("insufficient inventory")
    STOCK[sku] -= quantity
    return {"sku": sku, "available": STOCK[sku]}

def increment_available(sku: str, quantity: int) -> dict:
    STOCK[sku] = quantity_for(sku) + quantity
    return {"sku": sku, "available": STOCK[sku]}
