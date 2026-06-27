def anthropic_error(message: str, err_type: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}}
