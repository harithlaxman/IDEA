from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Allowlist ---------------------------------------------------------------

# Lowercased here, and `is_allowed` lowercases the incoming address: Google
# returns the email lowercased, so a mixed-case entry would silently 403.
ALLOWED_EMAILS: set[str] = {e.lower() for e in {
    "vitaliy@vt.edu",
    "naren@cs.vt.edu",
    "raquib@vt.edu",
    "eson@vt.edu",
    "shambhavi@vt.edu",
    "nwself@vt.edu",
    "brian.oleary@chronicle.com",
    "chris.contakes@chronicle.com",
    "Dan.Berrett@chronicle.com",
    "David.Wescott@chronicle.com",
    "jacquelyn.elias@chronicle.com",
    "sylvain.nazef@chronicle.com",
    "venketesh.subramony@chronicle.com",
    "Ron.Coddington@chronicle.com",
    "Michael.riley@chronicle.com",
    "venketeshs@gmail.com",
    "mgriley@gmail.com",
    "ccontakes@gmail.com",
    "roncoddington@gmail.com",
    "hatchjt@gmail.com",
    "ergoldstein@gmail.com",
    "sathappan.spm@gmail.com",
    "baiocchiaisha@gmail.com",
    "dankberrett@gmail.com",
    "meganzahneis@gmail.com",
    "fcdiep@gmail.com",
    "tswaak27@gmail.com",
    "harithlaxman@gmail.com",
    "harithlaxman@vt.edu",
    "matt.lyndaker@chronicle.com"
}}

# --- Auth --------------------------------------------------------------------

# No fallback: a committed default would be a publicly known signing key, so a
# misconfigured deploy should fail at import rather than forge sessions.
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set. It signs session JWTs and the session cookie "
        "and has no default. Set the JWT_SECRET environment variable (ECS task "
        "definition in deployed environments, .env locally) to a random secret, "
        "e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
    )

JWT_ALG = "HS256"
JWT_TTL_SECONDS = 60 * 60 * 8  # 8h

COOKIE_NAME = "dw_session"
COOKIE_MAX_AGE = JWT_TTL_SECONDS

# Secure by default. Set COOKIE_SECURE=0 for local http, where the browser
# drops a secure cookie and login silently fails.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "1") == "1"

# Google OAuth client credentials.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback"
)

# Enables /auth/dev-login, which accepts any allowlisted email without Google.
# Off by default so a forgotten variable can't leave that endpoint open.
DEV_AUTH_BYPASS = os.getenv("DEV_AUTH_BYPASS", "0") == "1"

# --- Database ----------------------------------------------------------------

def _ensure_asyncpg(url: str) -> str:
    """Normalize a Postgres URL onto the asyncpg driver.

    A bare "postgresql://" defaults to psycopg2, which isn't a dependency, and
    managed hosts hand out "postgres://", which SQLAlchemy rejects outright.
    Already-qualified URLs pass through untouched.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


MAIN_DATABASE_URL = _ensure_asyncpg(os.getenv("MAIN_DATABASE_URL", ""))
IPEDS_DATABASE_URL = os.getenv("IPEDS_DATABASE_URL", "")

CHECKPOINT_DATABASE_URL = os.getenv(
    "CHECKPOINT_DATABASE_URL",
    MAIN_DATABASE_URL.replace("+asyncpg", "") if MAIN_DATABASE_URL else "",
)

AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic:claude-sonnet-4-5")

# Denominator for the context meter under the composer.
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "200000"))

# Token count at which ContextEditingMiddleware starts clearing tool results.
CLEAR_TOOL_USES_TRIGGER = int(os.getenv("CLEAR_TOOL_USES_TRIGGER", "30000"))

# Graph steps one turn may take before LangGraph raises GraphRecursionError.
# A tool call costs two (model, then tool), so this is ~17 of them. Well under
# LangGraph's own 10007 default: the cap is here to stop a looping agent from
# billing indefinitely, not to accommodate a long analysis.
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "50"))

# --- Paths -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

APP_VERSION = "v0.2.0-agents"
