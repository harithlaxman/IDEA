"""FastAPI dependencies + request-scoped resolution helpers.

The one real dependency is `CurrentUser` (JWT cookie -> DB-backed `User`).
`optional_user_email` / `require_user_email` expose just the email for routes
that allow anonymous access or manage their own DB session; `load_owned_session`
/ `owned_session_id` gate session access on ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import verify_jwt
from .config import COOKIE_NAME
from .db import get_db
from .models import User
from .store import Session, store


def _unauthenticated(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


# --- Auth: cookie -> email --------------------------------------------------


def _email_from_request(request: Request) -> str | None:
    """Resolve the JWT cookie to an email, or None if absent/invalid.

    The single home of the cookie -> verify_jwt -> email chain.
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = verify_jwt(token)
    if not payload:
        return None
    return payload.get("email") or None


def optional_user_email(request: Request) -> str | None:
    """The authenticated user's email, or None — for routes that redirect.

    Full-page routes send an unauthenticated visitor to /login rather than
    returning 401 JSON.
    """
    return _email_from_request(request)


def require_user_email(request: Request) -> str:
    """The authenticated user's email, or 401."""
    email = _email_from_request(request)
    if not email:
        raise _unauthenticated("Not authenticated")
    return email


# --- Auth: cookie -> DB-backed user ----------------------------------------


@dataclass
class UserContext:
    """Authenticated user + owned DB session, resolved from the JWT cookie."""

    db: AsyncSession
    user: User


async def current_user_ctx(
    request: Request, db: AsyncSession = Depends(get_db)
) -> UserContext:
    """Resolve the JWT email to a User row; return (db, user).

    The row is created on first login, so it exists for any allowlisted email.
    """
    email = require_user_email(request)
    user = await store.get_or_create_user(db, email)
    return UserContext(db=db, user=user)


CurrentUser = Annotated[UserContext, Depends(current_user_ctx)]


# --- Session access ---------------------------------------------------------


async def load_owned_session(ctx: UserContext, session_id: str) -> Session:
    """The user's session with its messages loaded, or 404."""
    sess = await store.get_session(ctx.db, ctx.user.id, session_id)
    if not sess:
        raise _not_found("Session not found")
    return sess


async def owned_session_id(ctx: UserContext, session_id: str) -> str:
    """Canonical id of the session, or 404 if the user doesn't own it.

    Ownership only: loading the transcript here would put the whole
    conversation, which grows without bound, on every turn's critical path.
    """
    owned = await store.owned_session_ids(ctx.db, ctx.user.id, [session_id])
    if not owned:
        raise _not_found("Session not found")
    return owned[0]
