from backend.app.utils.jwt_tokens import decode_access_token

def require_authenticated_user(authorization: str) -> str:
    if not authorization.startswith("Bearer "):
        raise PermissionError("missing bearer token")
    claims = decode_access_token(authorization.removeprefix("Bearer "))
    if claims.get("expired"):
        raise PermissionError("expired session")
    return claims["sub"]
