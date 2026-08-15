from backend.app.repositories import user_repository
from backend.app.utils import jwt_tokens

def authenticate_user(credentials: dict) -> dict:
    user = user_repository.find_by_email(credentials["email"])
    if not user or not user_repository.verify_password(user, credentials["password"]):
        raise ValueError("invalid credentials")
    return jwt_tokens.issue_token_pair(user["id"])

def rotate_session(refresh_token: str) -> dict:
    claims = jwt_tokens.decode_refresh_token(refresh_token)
    return jwt_tokens.issue_token_pair(claims["sub"])
