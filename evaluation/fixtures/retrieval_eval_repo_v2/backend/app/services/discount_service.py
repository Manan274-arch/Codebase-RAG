from decimal import Decimal

def final_total(items: list[dict], coupon: str | None) -> Decimal:
    subtotal = sum(Decimal(str(item["price"])) * item["quantity"] for item in items)
    discount = Decimal("0.10") if coupon == "SAVE10" else Decimal("0")
    return subtotal * (Decimal("1") - discount)

def preview_total(items: list[dict]) -> Decimal:
    return final_total(items, None)
