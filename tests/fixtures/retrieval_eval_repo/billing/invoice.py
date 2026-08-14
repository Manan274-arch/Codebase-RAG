from decimal import Decimal


def calculate_invoice_total(line_items: list[Decimal]) -> Decimal:
    """Calculate the invoice total from every billed line item."""
    return sum(line_items, start=Decimal("0"))


def mark_payment_failed(invoice_id: str, reason: str) -> dict[str, str]:
    """Record a failed payment so billing can retry it later."""
    return {"invoice_id": invoice_id, "status": "failed", "reason": reason}
