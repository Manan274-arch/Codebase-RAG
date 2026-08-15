PROCESSED_KEYS: set[str] = set()

def claim_once(key: str) -> bool:
    if key in PROCESSED_KEYS:
        return False
    PROCESSED_KEYS.add(key)
    return True
