class FakeInventoryRepository:
    def decrement_available(self, sku: str, quantity: int) -> dict:
        return {"sku": sku, "available": 999 - quantity}
