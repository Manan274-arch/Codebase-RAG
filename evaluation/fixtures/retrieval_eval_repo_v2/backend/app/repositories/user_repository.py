USERS: dict[str, dict] = {}

def find_by_email(email: str) -> dict | None:
    return next((user for user in USERS.values() if user["email"] == email), None)

def verify_password(user: dict, password: str) -> bool:
    return user.get("password_hash") == f"hashed:{password}"

def get_public_profile(user_id: str) -> dict:
    user = USERS[user_id]
    return {"id": user_id, "name": user["name"], "timezone": user["timezone"]}

def update_profile(user_id: str, changes: dict) -> dict:
    USERS[user_id].update(changes)
    return get_public_profile(user_id)
