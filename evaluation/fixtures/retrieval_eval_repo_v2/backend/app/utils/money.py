from decimal import Decimal, ROUND_HALF_UP

def round_currency(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def format_currency(amount: Decimal, currency: str = "USD") -> str:
    return f"{currency} {round_currency(amount)}"
