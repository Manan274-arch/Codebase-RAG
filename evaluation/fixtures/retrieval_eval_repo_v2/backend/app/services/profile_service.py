from backend.app.repositories import user_repository

def load_profile(user_id: str) -> dict:
    return user_repository.get_public_profile(user_id)

def update_profile(user_id: str, changes: dict) -> dict:
    allowed = {key: value for key, value in changes.items() if key in {"name", "timezone"}}
    return user_repository.update_profile(user_id, allowed)
