from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import cookie_params, dev_bypass_enabled, is_allowed, issue_jwt, oauth
from ..config import GOOGLE_REDIRECT_URI
from ..db import get_db
from ..store import store

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth, or fall back to the dev login page."""
    # `oauth.google` raises AttributeError when the client was never registered,
    # so ask the registry instead — create_client() returns None in that case.
    google = oauth.create_client("google")
    if google is not None and not dev_bypass_enabled():
        redirect_uri = GOOGLE_REDIRECT_URI
        return await google.authorize_redirect(request, redirect_uri)
    # Dev bypass: show the local login picker.
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@router.get("/callback")
async def callback(request: Request, db: AsyncSession = Depends(get_db)):
    """OAuth callback — exchange code, allowlist check, persist user, issue JWT."""
    token = await oauth.google.authorize_access_token(request)
    user = token.get("userinfo") or {}
    email = user.get("email", "")
    if not email or not is_allowed(email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )
    name = user.get("name")  # from the `profile` scope
    await store.get_or_create_user(db, email, name)
    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    resp.set_cookie(value=issue_jwt(email), **cookie_params())
    return resp


@router.post("/dev-login")
async def dev_login(payload: dict, db: AsyncSession = Depends(get_db)):
    """Dev-only: accept an allowlisted email and issue the same JWT, no Google.

    404s unless DEV_AUTH_BYPASS is on — that flag alone gates this endpoint.
    """
    if not dev_bypass_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dev bypass disabled"
        )
    email = (payload.get("email") or "").strip().lower()
    if not email or not is_allowed(email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Email not on allowlist"
        )
    await store.get_or_create_user(db, email, name=None)
    resp = Response()
    resp.set_cookie(value=issue_jwt(email), **cookie_params())
    return resp


@router.post("/logout")
async def logout():
    resp = Response()
    resp.delete_cookie(key=cookie_params()["key"], path="/")
    return resp
