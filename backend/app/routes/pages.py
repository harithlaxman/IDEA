"""htmx-driven page routes. Returns HTML fragments, not JSON.

``/`` renders the full layout; later interactions swap fragments via hx-target.
The query SSE stream lives in sessions.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..turn import context_usage, get_thread_config, pending_interrupt
from ..auth import dev_bypass_enabled
from ..config import FRONTEND_DIR, GOOGLE_CLIENT_ID
from ..db import async_session
from ..deps import (
    CurrentUser,
    UserContext,
    load_owned_session,
    optional_user_email,
)
from ..store import ordered_messages, preference_items, store

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory=str(FRONTEND_DIR))


# --- App shell --------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    email = optional_user_email(request)
    if not email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    # Authed — resolve user + sessions via DB.
    async with async_session() as db:
        user = await store.get_or_create_user(db, email)
        sessions = await store.list_session_summaries(db, user.id)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"email": user.email, "sessions": sessions},
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If already authed, go home.
    if optional_user_email(request):
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "dev_bypass": dev_bypass_enabled(),
            "oauth_configured": bool(GOOGLE_CLIENT_ID),
        },
    )


# --- Sidebar fragments ------------------------------------------------------


@router.get("/frag/sessions", response_class=HTMLResponse)
async def frag_sessions(request: Request, ctx: CurrentUser):
    sessions = await store.list_session_summaries(ctx.db, ctx.user.id)
    return templates.TemplateResponse(
        request, "fragments/sessions.html", {"sessions": sessions}
    )


@router.get("/frag/session/{session_id}", response_class=HTMLResponse)
async def frag_session(session_id: str, request: Request, ctx: CurrentUser):
    sess = await load_owned_session(ctx, session_id)
    messages = [m.to_dict() for m in ordered_messages(sess.messages)]

    config = get_thread_config(sess.id)
    agent = request.app.state.agent
    pending = await pending_interrupt(config=config, agent=agent)
    # The composer's context meter, restored from the checkpoint: a reload or a
    # session switch has no stream to read it off.
    usage = await context_usage(agent, config)
    return templates.TemplateResponse(
        request,
        "fragments/main.html",
        {
            "session": sess.to_summary(),
            "messages": messages,
            # The pause the checkpoint says this thread is on, if any: a
            # clarify is answered by typing, a confirm by clicking. main.html
            # restores whichever control the answer needs.
            "pending": (
                {
                    "kind": "clarify" if pending.is_clarify else "confirm",
                    "text": pending.question
                    if pending.is_clarify
                    else pending.preference,
                }
                if pending
                else None
            ),
            "usage": usage,
        },
    )


# --- Delete modal fragment --------------------------------------------------


@router.get("/frag/delete-history", response_class=HTMLResponse)
async def frag_delete_history(request: Request, ctx: CurrentUser):
    sessions = await store.list_session_summaries(ctx.db, ctx.user.id)
    return templates.TemplateResponse(
        request, "fragments/delete_history.html", {"sessions": sessions}
    )


# --- Rate modal fragment ----------------------------------------------------


@router.get("/frag/rate/{session_id}", response_class=HTMLResponse)
async def frag_rate(session_id: str, request: Request, ctx: CurrentUser):
    sess = await load_owned_session(ctx, session_id)
    messages = ordered_messages(sess.messages)
    latest_asst = next((m for m in reversed(messages) if m.role == "assistant"), None)
    message = {"message": latest_asst.content, "id": latest_asst.id} if latest_asst else None
    return templates.TemplateResponse(
        request,
        "fragments/rate.html",
        {"session_id": session_id, "message": message},
    )


# --- Preferences modal fragment ---------------------------------------------


@router.get("/frag/preferences", response_class=HTMLResponse)
async def frag_preferences(request: Request, ctx: CurrentUser):
    # ctx.user is the ORM row current_user_ctx already loaded, so reading the
    # column here costs nothing extra. Writes still go through the store.
    return templates.TemplateResponse(
        request,
        "fragments/preferences.html",
        {"items": preference_items(ctx.user.preferences)},
    )
