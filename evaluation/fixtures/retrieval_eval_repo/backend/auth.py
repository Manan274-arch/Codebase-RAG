from datetime import datetime


def validate_refresh_token(refresh_token: str, expires_at: datetime) -> bool:
    """Reject an expired refresh token before issuing new credentials."""
    if expires_at <= datetime.now():
        raise ValueError("refresh token expired")
    return refresh_token.startswith("refresh_")


def issue_access_token(user_id: str) -> str:
    return f"access_{user_id}"
