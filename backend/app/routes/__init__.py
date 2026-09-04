from .auth import router as auth_router
from .pages import router as pages_router
from .preferences import router as preferences_router
from .sessions import router as sessions_router

__all__ = ["auth_router", "pages_router", "preferences_router", "sessions_router"]
