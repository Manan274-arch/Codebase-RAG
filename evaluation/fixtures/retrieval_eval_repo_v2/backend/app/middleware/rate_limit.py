COUNTERS: dict[str, int] = {}

def enforce_request_limit(client_id: str, limit: int = 100) -> None:
    COUNTERS[client_id] = COUNTERS.get(client_id, 0) + 1
    if COUNTERS[client_id] > limit:
        raise RuntimeError("request rate exceeded")
