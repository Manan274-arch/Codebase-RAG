from backend.app.repositories.inventory_repository import quantity_for

def reconcile_catalog_stock(skus: list[str]) -> dict[str, int]:
    return {sku: quantity_for(sku) for sku in skus}
