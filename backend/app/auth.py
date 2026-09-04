"""Auth: JWT issue/verify + Google OAuth (Authlib) + dev bypass.

The email is encoded into the JWT; the allowlist lives in config. With
GOOGLE_CLIENT_ID/SECRET set the full OAuth round-trip runs; otherwise
/auth/dev-login issues the same JWT so local dev needs no OAuth setup.
"""

from __future__ import annotations

import time
from typing import Any

import jwt
from authlib.integrations.starlette_client import OAuth

from .config import (
    ALLOWED_EMAILS,
    COOKIE_MAX_AGE,
    COOKIE_NAME,
    COOKIE_SECURE,
    DEV_AUTH_BYPASS,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    JWT_ALG,
    JWT_SECRET,
    JWT_TTL_SECONDS,
)

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def issue_jwt(email: str) -> str:
    now = int(time.time())
    payload = {
        "sub": email,
        "email": email,
        "iat": now,
        "exp": now + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def verify_jwt(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None


def is_allowed(email: str) -> bool:
    return email.strip().lower() in ALLOWED_EMAILS


def cookie_params() -> dict:
    return {
        "key": COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": COOKIE_SECURE,  # HTTPS-only unless COOKIE_SECURE=0 (local http)
        "max_age": COOKIE_MAX_AGE,
        "path": "/",
    }



def dev_bypass_enabled() -> bool:
    return DEV_AUTH_BYPASS
