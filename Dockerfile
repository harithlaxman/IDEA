FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /bin/uv

WORKDIR /app

# Create the runtime user up front and give it /app, so the venv and source are
# written with the right ownership. A recursive chown afterwards would rewrite
# every venv file into a new layer, roughly doubling the image's Python footprint.
RUN useradd --create-home --uid 10001 appuser \
 && chown appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1

# Dependencies before source: editing backend/ or frontend/ must not invalidate
# this layer.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# entrypoint.sh is mode 100755 in git and COPY preserves the exec bit, so no
# chmod is needed here.
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser entrypoint.sh ./entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
