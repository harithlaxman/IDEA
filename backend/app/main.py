from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from .agent import build_agent
from .config import (
    APP_VERSION,
    CHECKPOINT_DATABASE_URL,
    COOKIE_SECURE,
    JWT_SECRET,
    STATIC_DIR,
)
from .db import create_schema
from .routes import auth_router, pages_router, preferences_router, sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT_DATABASE_URL:
        raise RuntimeError("CHECKPOINT_DATABASE_URL not configured")

    # The app's own tables; checkpointer.setup() below does LangGraph's.
    await create_schema()

    async with AsyncConnectionPool(
        conninfo=CHECKPOINT_DATABASE_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
        max_size=10,
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        app.state.checkpointer = checkpointer
        app.state.agent = build_agent(checkpointer)
        yield
    # pool closed here on shutdown


class RevalidatedStaticFiles(StaticFiles):
    """StaticFiles that makes a browser ask before reusing a cached asset.

    Starlette sends an ETag and Last-Modified but no Cache-Control, which
    leaves the browser free to invent a freshness lifetime and serve app.js
    from cache without checking. A shipped frontend change then lands on a page
    still running the old script — server-rendered markup from the new
    templates driven by JS that has never heard of it, which looks like a
    feature that half works rather than like a stale file.

    `no-cache` does not disable caching, it only requires the revalidation, so
    the ETag still turns an unchanged asset into a 304 with no body.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        # Set on the 304 too: NotModifiedResponse keeps cache-control among the
        # handful of headers it forwards.
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="IDEA", version=APP_VERSION, lifespan=lifespan)

    # Authlib needs this to persist the OAuth state token across the
    # redirect → callback round-trip (CSRF protection).
    app.add_middleware(
        SessionMiddleware, secret_key=JWT_SECRET, https_only=COOKIE_SECURE
    )

    # Static assets (htmx, extensions, app.js, styles.css)
    app.mount(
        "/static", RevalidatedStaticFiles(directory=str(STATIC_DIR)), name="static"
    )

    # Routers
    app.include_router(pages_router)
    app.include_router(auth_router)
    app.include_router(sessions_router)
    app.include_router(preferences_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": APP_VERSION}

    return app


app = create_app()
