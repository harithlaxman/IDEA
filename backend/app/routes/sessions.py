from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..turn import process_query
from ..db import async_session
from ..deps import CurrentUser, UserContext, owned_session_id
from ..store import preference_items, store

router = APIRouter(tags=["sessions"])


def sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/sessions")
async def create_session(
    ctx: CurrentUser, payload: dict[str, Any] | None = None
):
    title = (payload or {}).get("title", "New Analysis")
    sess = await store.create_session(ctx.db, ctx.user.id, title=title)
    return sess.to_summary()


@router.delete("/sessions")
async def delete_sessions(
    payload: dict, request: Request, ctx: CurrentUser
):
    """Delete sessions and the LangGraph checkpoint thread behind each one.

    The agent's working memory lives in the checkpoint tables under
    `thread_id == str(session.id)`
    """
    ids = payload.get("ids", [])
    owned = await store.owned_session_ids(ctx.db, ctx.user.id, ids)
    if not owned:
        return {"deleted": []}
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer unavailable — Cannot delete a session without purging its checkpoints.",
        )
    for sid in owned:
        await checkpointer.adelete_thread(sid)
    await store.delete_sessions(ctx.db, ctx.user.id, owned)
    return {"deleted": owned}


async def _turn_stream(
    events, message: str, session_id: str, user_id, *, no_final_error: str
):
    """Relay one agent turn as SSE, persisting the user+assistant pair on final.

    Stream shape:
        event: status  data: {"text": "..."}
        event: usage   data: {"tokens": 0, "window": 200000, "exact": true}
        event: text    data: {"seq": 0, "delta": "..."}
        event: tool    data: {"phase": "start"|"end", "name": "...", "run_id": "..."}
        event: clarify data: {"question": "...", "message_id": "...", "step": 0}
        event: confirm data: {"preference": "...", "message_id": "...", "step": 0}
        event: final   data: {"text": "...", "timeline": [...], "artifacts": [...],
                               "error": null, "message": "...", "message_id": "...", ...}
    ``status`` and ``usage`` are transient — neither is persisted, and the
    context meter is derived from the checkpoint, not from the messages table.
    ``text``/``tool`` build the persisted timeline that ``seq`` indexes. ``final.text`` is the answer alone, not every segment
    concatenated, and ``message_id`` names the assistant message of the pair.

    Opens its own DB session — FastAPI closes the request-scoped one before the
    StreamingResponse body is sent.
    """

    async def persist(final: dict) -> dict:
        async with async_session() as db:
            user_msg, asst_msg = await store.add_turn(
                db, user_id, session_id, message, final
            )
        return {
            "text": final.get("text", ""),
            "artifacts": final.get("artifacts", []),
            "timeline": final.get("timeline", []),
            "error": final.get("error"),
            "message": user_msg.content,
            "message_id": asst_msg.id,
            "failed": asst_msg.failed,
            "step": asst_msg.step,
        }

    final = None
    paused = False
    async for event in events:
        et = event.get("type")
        if et == "status":
            yield sse("status", {"text": event.get("text", "")})
        elif et == "usage":
            yield sse(
                "usage",
                {
                    "tokens": event.get("tokens", 0),
                    "window": event.get("window", 0),
                    "exact": event.get("exact", False),
                },
            )
        elif et == "text":
            yield sse(
                "text",
                {
                    "seq": event.get("seq", 0),
                    "delta": event.get("delta", ""),
                },
            )
        elif et == "tool":
            yield sse(
                "tool",
                {
                    "phase": event.get("phase"),
                    "name": event.get("name"),
                    "run_id": event.get("run_id"),
                    "ok": event.get("ok", True),
                },
            )
        elif et == "clarify":
            # Persisted like any other turn: the question is the answer, and
            # the event's timeline already carries the rest of the turn.
            paused = True
            question = event.get("question", "")
            payload = await persist(
                {
                    "text": question,
                    "timeline": event.get("timeline", []),
                    "artifacts": [],
                    "error": None,
                }
            )
            yield sse(
                "clarify",
                {
                    "question": question,
                    "message_id": payload["message_id"],
                    "step": payload["step"],
                },
            )
        elif et == "confirm":
            # The other pause shape. Same persistence rule as clarify — the
            # timeline already ends with the proposal — but the answer is a
            # click, so the client gets the preference text to render buttons
            # against rather than a question to type under.
            paused = True
            preference = event.get("preference", "")
            timeline = event.get("timeline", [])
            payload = await persist(
                {
                    "text": timeline[-1]["text"] if timeline else preference,
                    "timeline": timeline,
                    "artifacts": [],
                    "error": None,
                }
            )
            yield sse(
                "confirm",
                {
                    "preference": preference,
                    "message_id": payload["message_id"],
                    "step": payload["step"],
                },
            )
        elif et == "final":
            final = event
            yield sse("final", await persist(final))
    if final is None and not paused:
        yield sse(
            "final",
            await persist(
                {
                    "text": "",
                    "artifacts": [],
                    "timeline": [],
                    "error": no_final_error,
                }
            ),
        )


def _stream_response(gen) -> StreamingResponse:
    # A frame per token, so anything that buffers the body holds back the
    # whole answer and undoes the streaming.
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


DECISION_MESSAGES = {"approve": "Approved.", "reject": "Declined."}


def _require_message(payload: dict) -> str:
    """The user's text for this turn, or the stand-in for an approve/reject.

    Answering an approval gate is a click, so there is no typed message to
    require — but the pair still persists like any other turn, and an empty
    user bubble would read as a bug in the transcript.
    """
    decision = payload.get("decision")
    if decision in DECISION_MESSAGES:
        return DECISION_MESSAGES[decision]
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="message required"
        )
    return message


# Whether a thread is paused isn't the client's call to make, and a wrong guess
# strands it on an unanswered ask_user. process_query reads the interrupt from
# the checkpoint and picks the input shape itself.
@router.post("/sessions/{session_id}/query")
async def query(
    session_id: str, payload: dict, request: Request, ctx: CurrentUser
):
    """Run one agent turn over SSE, persisting a user+assistant pair on final."""
    sid = await owned_session_id(ctx, session_id)
    message = _require_message(payload)
    return _stream_response(
        _turn_stream(
            process_query(
                message,
                sid,
                str(ctx.user.id),
                request.app.state.agent,
                preferences=preference_items(ctx.user.preferences),
                decision=payload.get("decision"),
            ),
            message,
            sid,
            ctx.user.id,
            no_final_error="Backend returned no final event.",
        )
    )


@router.post("/sessions/{session_id}/messages/{message_id}/feedback")
async def feedback(
    session_id: str, message_id: str, payload: dict, ctx: CurrentUser
):
    fb = {
        "rating": int(payload.get("rating", 0)),
        "issue_type": payload.get("issue_type", "None"),
        "comment": (payload.get("comment") or "").strip(),
    }
    # No ownership pre-check: set_feedback joins through to the owning user.
    msg = await store.set_feedback(ctx.db, ctx.user.id, session_id, message_id, fb)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )
    return {"feedback": fb}
