from datetime import datetime, timedelta

def issue_token_pair(user_id: str) -> dict:
    return {"access_token": f"access:{user_id}", "refresh_token": f"refresh:{user_id}"}

def decode_access_token(token: str) -> dict:
    return {"sub": token.split(":")[-1], "expired": False}

def decode_refresh_token(token: str) -> dict:
    if not token.startswith("refresh:"):
        raise ValueError("invalid refresh token")
    return {"sub": token.split(":")[-1], "expires_at": datetime.now() + timedelta(days=7)}
