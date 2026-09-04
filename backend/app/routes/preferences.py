"""The user's saved preferences as JSON — read and replaced by the edit modal.

Its own module because pages.py returns HTML only and sessions.py is scoped to
a session, while preferences hang off the user.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import CurrentUser
from ..store import preference_items, store

router = APIRouter(tags=["preferences"])


@router.put("/preferences")
async def put_preferences(payload: dict, ctx: CurrentUser):
    """Replace the whole list. The modal edits a textarea, one item per line."""
    items = payload.get("items")
    if items is None:
        items = (payload.get("text") or "").splitlines()
    if not isinstance(items, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="items must be a list"
        )
    saved = await store.set_preferences(ctx.db, ctx.user.id, items)
    if saved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return {"items": saved}


@router.get("/preferences")
async def get_preferences(ctx: CurrentUser):
    return {"items": preference_items(ctx.user.preferences)}
