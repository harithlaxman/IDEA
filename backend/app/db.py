from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import MAIN_DATABASE_URL
from .models import Base

engine = (
    create_async_engine(MAIN_DATABASE_URL, echo=False) if MAIN_DATABASE_URL else None
)
async_session = (
    async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    if engine
    else None
)


async def create_schema() -> None:
    """Create any of the app's tables that don't exist yet.

    Idempotent, but CREATE-only: adding a column to an existing table needs a
    hand-written ALTER (or a dev DB rebuild).
    """
    if engine is None:
        raise RuntimeError("MAIN_DATABASE_URL not configured — cannot create schema")

    async with engine.begin() as conn:
        # create_all reflects then creates non-atomically, so the lock keeps
        # two replicas from creating the same table at once.
        await conn.execute(text("SELECT pg_advisory_xact_lock(920415)"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Yield an async DB session; raise if DATABASE_URL is not configured."""
    if async_session is None:
        raise RuntimeError("DATABASE_URL not configured — cannot access database")
    async with async_session() as session:
        yield session
