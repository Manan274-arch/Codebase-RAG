def deliver_notification(job: dict) -> dict:
    return {"delivered": True, "template": job["job"], "recipient": job.get("user_id")}
