"""Session/message store, backed by Postgres via SQLAlchemy async.

Every method takes a db session and user_id; ownership is verified per access.
Routes and templates get DTOs (Message, Session), so Jinja2's sync rendering
never touches an ORM lazy-loaded relationship.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Message as MessageORM
from .models import Session as SessionORM
from .models import User as UserORM

# --- DTOs (view models) ------------------------------------------------------


@dataclass
class Message:
    id: str
    step: int = 1
    role: str = "user"
    content: str = ""
    artifacts: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    error: str | None = None
    failed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    feedback: dict | None = None

    @property
    def log_entries(self) -> list:
        """The timeline minus its trailing text entry, which `content` holds.

        app.js applies the same rule live, so a card looks identical after a
        reload. A failed turn promoted nothing to `content`, so all of its
        timeline is log content — dropping the last entry would lose that prose.
        """
        if self.error:
            return self.timeline
        if self.timeline and self.timeline[-1].get("kind") == "text":
            return self.timeline[:-1]
        return self.timeline

    @property
    def tool_count(self) -> int:
        # Everything that isn't text, as main.html draws it — which also counts
        # pre-timeline rows (bare {"name", "ok"}, no "kind") correctly.
        return sum(1 for e in self.timeline if e.get("kind") != "text")

    def to_dict(self) -> dict:
        # asdict() sees only fields, so the two derived views are carried
        # explicitly for the templates that render from this dict.
        return {
            **asdict(self),
            "log_entries": self.log_entries,
            "tool_count": self.tool_count,
        }


@dataclass
class Session:
    id: str
    title: str = "New Analysis"
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_summary(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at,
        }


# --- ORM → DTO mapping -------------------------------------------------------


def _msg_to_dto(m: MessageORM) -> Message:
    return Message(
        id=str(m.id),
        step=m.step,
        role=m.role,
        content=m.content or "",
        artifacts=m.artifacts or [],
        timeline=m.timeline or [],
        error=m.error,
        failed=m.failed,
        timestamp=m.created_at.isoformat() if m.created_at else _now_iso(),
        feedback=m.feedback,
    )


def _sess_to_dto(s: SessionORM) -> Session:
    return Session(
        id=str(s.id),
        title=s.title,
        messages=[_msg_to_dto(m) for m in (s.messages or [])],
        created_at=s.created_at.isoformat() if s.created_at else "",
        updated_at=s.updated_at.isoformat() if s.updated_at else "",
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

MAX_PREFERENCES = 50
MAX_PREFERENCE_LEN = 200


def preference_items(raw: dict | None) -> list[str]:
    """The saved preference sentences in `users.preferences`, normalized.

    One shared normalizer for the three writers/readers (the edit UI, the
    save_user_preference tool, the prompt builder) so a list that round-trips
    through any of them comes back the same. Blanks and non-strings drop out,
    and a case-insensitive dedupe keeps the first spelling of a repeat.
    """
    items = (raw or {}).get("items") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()[:MAX_PREFERENCE_LEN]
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out[:MAX_PREFERENCES]


def _to_uuid(value: str) -> uuid.UUID | None:
    """Parse a string to UUID; return None on failure."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def ordered_messages(messages: list[Message]) -> list[Message]:
    """Messages in turn order: (step, timestamp).

    The ORM relationship already applies this ordering, but making it
    explicit keeps callers from depending on that side effect.
    """
    return sorted(messages, key=lambda m: (m.step, m.timestamp))

# --- Store -------------------------------------------------------------------


class Store:
    """DB-backed. Stateless — all state lives in Postgres."""

    # --- users --------------------------------------------------------------

    async def get_or_create_user(
        self, db: AsyncSession, email: str, name: str | None = None
    ) -> UserORM:
        """Return the existing user for `email`, or insert a new row if absent."""
        user = (
            await db.execute(select(UserORM).where(UserORM.email == email))
        ).scalar_one_or_none()
        if user is None:
            user = UserORM(email=email, name=name)
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    async def set_preferences(
        self, db: AsyncSession, user_id: uuid.UUID, items: list[str]
    ) -> list[str] | None:
        """Replace the user's whole preference list; return what was stored."""
        user = (
            await db.execute(select(UserORM).where(UserORM.id == user_id))
        ).scalar_one_or_none()
        if user is None:
            return None
        cleaned = preference_items({"items": items})
        # Whole-value assign: plain JSONB is not mutation-tracked, so editing
        # the dict in place leaves the attribute clean and commits nothing.
        user.preferences = {"items": cleaned}
        await db.commit()
        await db.refresh(user)
        return cleaned

    async def add_preference(
        self, db: AsyncSession, user_id: uuid.UUID, text: str
    ) -> list[str] | None:
        """Append one preference. A duplicate is absorbed by the normalizer."""
        user = (
            await db.execute(select(UserORM).where(UserORM.id == user_id))
        ).scalar_one_or_none()
        if user is None:
            return None
        return await self.set_preferences(
            db, user_id, [*preference_items(user.preferences), text]
        )

    # --- sessions -----------------------------------------------------------

    async def list_session_summaries(
        self, db: AsyncSession, user_id: uuid.UUID
    ) -> list[dict[str, str]]:
        """The sidebar's view of every session: id, title, when."""
        result = await db.execute(
            select(SessionORM.id, SessionORM.title, SessionORM.updated_at)
            .where(SessionORM.user_id == user_id)
            .order_by(SessionORM.updated_at.desc())
        )
        return [
            {
                "id": str(sid),
                "title": title,
                "updated_at": updated_at.isoformat() if updated_at else "",
            }
            for sid, title, updated_at in result.tuples().all()
        ]

    async def create_session(
        self, db: AsyncSession, user_id: uuid.UUID, title: str = "New Analysis"
    ) -> Session:
        sess = SessionORM(user_id=user_id, title=title)
        db.add(sess)
        await db.commit()
        # Re-fetch eagerly: refresh() alone leaves relationships lazy-loaded,
        # which raises MissingGreenlet in async context.
        result = await db.execute(
            select(SessionORM)
            .options(selectinload(SessionORM.messages))
            .where(SessionORM.id == sess.id)
        )
        sess = result.scalar_one()
        return _sess_to_dto(sess)

    async def get_session(
        self, db: AsyncSession, user_id: uuid.UUID, session_id: str
    ) -> Session | None:
        sid = _to_uuid(session_id)
        if sid is None:
            return None
        result = await db.execute(
            select(SessionORM)
            .options(selectinload(SessionORM.messages))
            .where(SessionORM.id == sid, SessionORM.user_id == user_id)
        )
        sess = result.scalar_one_or_none()
        return _sess_to_dto(sess) if sess else None


    async def owned_session_ids(
        self, db: AsyncSession, user_id: uuid.UUID, ids: list[str]
    ) -> list[str]:
        """Canonical string ids of the sessions in `ids` that `user_id` owns."""
        uuids = [u for u in (_to_uuid(i) for i in ids) if u is not None]
        if not uuids:
            return []
        result = await db.execute(
            select(SessionORM.id).where(
                SessionORM.id.in_(uuids), SessionORM.user_id == user_id
            )
        )
        return [str(sid) for sid in result.scalars().all()]


    async def delete_sessions(
        self, db: AsyncSession, user_id: uuid.UUID, ids: list[str]
    ) -> None:
        uuids = [u for u in (_to_uuid(i) for i in ids) if u is not None]
        if not uuids:
            return
        result = await db.execute(
            select(SessionORM).where(
                SessionORM.id.in_(uuids), SessionORM.user_id == user_id
            )
        )
        for sess in result.scalars().all():
            await db.delete(sess)
        await db.commit()

    # --- messages -----------------------------------------------------------

    async def add_turn(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: str,
        message: str,
        final: dict,
    ) -> tuple[Message, Message]:
        """Persist a user+assistant message pair atomically; return both DTOs.

        ``final`` is the agent's final event dict ({text, timeline, artifacts,
        error}), which becomes the assistant message.
        """
        sid = _to_uuid(session_id)
        if sid is None:
            raise ValueError(f"Invalid session_id: {session_id}")
        sess = await self._get_owned_session(db, user_id, sid)
        if sess is None:
            raise ValueError(f"Session not found: {session_id}")

        # Next step = max existing step + 1 (or 1 if no messages).
        step_result = await db.execute(
            select(func.coalesce(func.max(MessageORM.step), 0) + 1).where(
                MessageORM.session_id == sid
            )
        )
        step = step_result.scalar_one()

        user_msg = MessageORM(
            session_id=sid,
            step=step,
            role="user",
            content=message,
        )
        asst_msg = MessageORM(
            session_id=sid,
            step=step,
            role="assistant",
            content=final.get("text", ""),
            artifacts=final.get("artifacts", []),
            timeline=final.get("timeline", []),
            error=final.get("error"),
            failed=bool(final.get("error")),
        )
        db.add(user_msg)
        db.add(asst_msg)

        # Derive title from first message if still default.
        if sess.title in ("", "New Analysis"):
            sess.title = message[:60] or "Analysis"
        sess.updated_at = func.now()

        # No refresh(): expire_on_commit=False plus eager_defaults (created_at
        # via RETURNING on flush) leaves both rows fully populated here.
        await db.commit()
        return _msg_to_dto(user_msg), _msg_to_dto(asst_msg)

    async def set_feedback(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        feedback: dict,
    ) -> Message | None:
        sid = _to_uuid(session_id)
        mid = _to_uuid(message_id)
        if sid is None or mid is None:
            return None
        result = await db.execute(
            select(MessageORM)
            .join(SessionORM, MessageORM.session_id == SessionORM.id)
            .where(
                MessageORM.id == mid,
                SessionORM.id == sid,
                SessionORM.user_id == user_id,
            )
        )
        msg = result.scalar_one_or_none()
        if msg is None:
            return None
        msg.feedback = feedback
        await db.commit()
        await db.refresh(msg)
        return _msg_to_dto(msg)

    # --- internal -----------------------------------------------------------

    async def _get_owned_session(
        self, db: AsyncSession, user_id: uuid.UUID, sid: uuid.UUID
    ) -> SessionORM | None:
        result = await db.execute(
            select(SessionORM).where(
                SessionORM.id == sid, SessionORM.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

store = Store()
