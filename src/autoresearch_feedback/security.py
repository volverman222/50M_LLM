from __future__ import annotations

import hashlib
import hmac


def verify_wandb_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not secret or not signature:
        return False
    supplied = signature.removeprefix("sha256=").strip().lower()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


def verify_bearer_token(authorization: str | None, token: str | None) -> bool:
    if token is None:
        return True
    if authorization is None:
        return False
    expected = f"Bearer {token}"
    return hmac.compare_digest(authorization, expected)
