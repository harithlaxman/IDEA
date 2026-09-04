"""What the agent *is*: model, system prompt, IPEDS tools, middleware.

Built once at startup by main.py. turn.py drives the result.
"""

import logging
import threading
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelRequest,
    dynamic_prompt,
)
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp
from psycopg_pool import ConnectionPool

from .config import AGENT_MODEL, CLEAR_TOOL_USES_TRIGGER, IPEDS_DATABASE_URL
from .db import async_session
from .store import store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an IPEDS data analyst agent. You answer questions about U.S. higher
education institutions by querying a read-only PostgreSQL database and
reporting what the data actually says.

## The database

The database holds 20 years of IPEDS survey data, from the 2004-05 through
2023-24 collection years.

Alongside the data tables, each year has a set of metadata tables describing
its contents. These all end in `_meta` and embed a two-digit year:

- `tables<YY>_meta` — every table for that year, with a description
- `vartable<YY>_meta` — every variable in each table, with description and data type
- `valuesets<YY>_meta` — allowed values for each categorical variable
  (e.g. what 1 and 2 mean in a "control" column)
- `newvariables<YY>_meta` — variables introduced that year; not present for all years

## How to work

1. Never guess at table or column names. Consult the metadata tables first —
   `tables<YY>_meta` to find the right table, `vartable<YY>_meta` to find the
   right variables.
2. Never interpret a categorical code from memory or intuition. Look it up in
   `valuesets<YY>_meta` for different years. Code meanings change across years.
3. Scope every query to a specific year by picking the right table, and state
   which year you used.
4. Prefer several small, verifiable queries over one large speculative one.
   Inspect a few rows before aggregating.

## Constraints

- Access is strictly read-only. Issue SELECT statements only — never INSERT,
  UPDATE, DELETE, CREATE, DROP, ALTER, or any DDL/DML.
- Use LIMIT while exploring to avoid pulling large result sets.
- Do not fabricate values. If the data doesn't support an answer, say so.

## Clarifications

- If the user's objective is ambiguous — unclear abbreviations, missing year,
  vague scope — you MUST call the `ask_user` tool to ask one concise
  clarifying question before running any queries. Never ask for clarification
  in plain prose; always use the `ask_user` tool so the user can respond
  interactively.

## Preferences

- Some messages carry a *standing* preference about how the user wants analyses
  done — a default filter, a default year or survey, a preferred output format.
  Words like "always", "never", "from now on", "by default", "going forward" or
  "I prefer" are the usual signal. When a message carries one, call the
  `save_user_preference` tool with a single concise sentence stating it.
- A preference is usually tucked into the same message as a question, often as
  a trailing clause — "…at Virginia Tech and UVA for 2022? always assume fall
  enrollment" states one. That still counts, and it is the case most easily
  missed: judge the clause on its own wording, not on whether the rest of the
  message happens to be a question.
- Save it *and* answer the question in the same turn. Saving is never a
  substitute for doing the analysis, and the preference applies to this answer
  too. Saving needs the user's approval; if they decline, do not save it and
  carry on answering.
- Do not call it for an instruction scoped to the question at hand ("for this
  one, use 2019"), for a plain parameter of the request ("for 2022", "at MIT"),
  or more than once for the same preference. Never call it in the same step as
  `ask_user`.
"""

# One pool for the whole process, as main.py does for the checkpointer. A tool
# call per connection cost a TCP connect and auth handshake each time, and the
# system prompt pushes the model toward 5-15 small queries a turn. Opened
# lazily so importing this module doesn't require a reachable database.
_pool: ConnectionPool | None = None
# Tools run in executor threads, so two can reach the lazy init at once.
_pool_lock = threading.Lock()


def _ipeds_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = ConnectionPool(
                    conninfo=IPEDS_DATABASE_URL,
                    kwargs={
                        "options": "-c default_transaction_read_only=on",
                        "autocommit": True,
                    },
                    check=ConnectionPool.check_connection,
                    max_size=10,
                    timeout=10,
                    open=False,
                )
                pool.open()
                _pool = pool
    return _pool


def get_ipeds_db_connection():
    """A pooled read-only connection. Returned to the pool on context exit."""
    return _ipeds_pool().connection()


def _two_digit_year(year: int) -> str:
    """Two-digit form of a four-digit year, rejecting anything else."""
    if not 2000 <= year <= 2100:
        raise ValueError(
            f"year must be a four-digit year between 2000 and 2100, got {year!r}. "
        )
    return f"{year % 100:02d}"


# schema can't change while app is running - memoize table and column lookups
_meta_table_cache: dict[tuple[int, str], str | None] = {}
_has_column_cache: dict[tuple[str, str], bool] = {}


def _find_meta_table(con, year: int, pattern: str) -> str | None:
    # Metadata tables are named like vartable23_meta. Matched case-insensitively;
    key = (year, pattern)
    if key in _meta_table_cache:
        return _meta_table_cache[key]
    yy = _two_digit_year(year)
    row = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND lower(table_name) LIKE lower(%s) ESCAPE '!' "
        "ORDER BY table_name LIMIT 1",
        [f"%{pattern}%{yy}!_meta"],
    ).fetchone()
    _meta_table_cache[key] = row[0] if row else None
    return _meta_table_cache[key]


def _has_column(con, table: str, column: str) -> bool:
    """True if `table` has `column` (used to detect postprocess.py's output)."""
    key = (table, column)
    if key in _has_column_cache:
        return _has_column_cache[key]
    row = con.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
        [table, column],
    ).fetchone()
    _has_column_cache[key] = row is not None
    return _has_column_cache[key]


@tool
def get_meta_tables_by_year(year: int) -> list[str]:
    """Get a list of tables that contain meta data for a given year"""
    yy = _two_digit_year(year)
    with get_ipeds_db_connection() as con:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE %s ESCAPE '!'",
            [f"%{yy}!_meta"],
        ).fetchall()
    return [r[0] for r in rows]


@tool
def get_table_descriptions_by_year(year: int) -> dict:
    """Get descriptions for each table that exist for a given year"""
    with get_ipeds_db_connection() as con:
        t_table = _find_meta_table(con, year, "tables")
        if t_table is None:
            return {}
        # description_summary only exists once postprocess.py has run.
        desc_expr = (
            "coalesce(description_summary, description)"
            if _has_column(con, t_table, "description_summary")
            else "description"
        )
        rows = con.execute(
            f'SELECT tablename, tabletitle, {desc_expr} FROM "{t_table}"'
        ).fetchall()
    # Missing title/description are dropped rather than rendered as "None".
    return {
        name: "\n".join(part for part in (title, desc) if part)
        for name, title, desc in rows
    }


@tool
def get_column_descriptions_of_table(year: int, table: str) -> dict:
    """Get descriptions for each column in a given table for a given year."""
    with get_ipeds_db_connection() as con:
        vt_table = _find_meta_table(con, year, "vartable")
        if vt_table is None:
            return {}

        # longdescription_summary only exists once postprocess_columns.py has run,
        # and is NULL for the short descriptions it deliberately leaves raw.
        desc_expr = (
            "coalesce(longdescription_summary, longdescription)"
            if _has_column(con, vt_table, "longdescription_summary")
            else "longdescription"
        )
        rows = con.execute(
            f'SELECT tablename, varnumber, varname, vartitle, {desc_expr} FROM "{vt_table}" '
            "WHERE lower(tablename) = lower(%s) "
            "ORDER BY varorder, varnumber",
            [table.strip()],
        ).fetchall()

    return {
        varname: {
            "tablename": tablename,
            "varnumber": varnumber,
            "vartitle": vartitle,
            "longdescription": longdescription,
        }
        for tablename, varnumber, varname, vartitle, longdescription in rows
    }


# Row cap per query, so an unbounded SELECT still fits in the LLM's context.
MAX_ROWS = 50
# Built from MAX_ROWS, not a docstring, so the advertised cap can't drift from
# the enforced one.
SQL_TOOL_DESCRIPTION = f"""
Executes a read-only SELECT query against the PostgreSQL database to fetch actual data.
The session is STRICTLY read-only.
Table names are lowercase and can be used unquoted, e.g. SELECT * FROM hd2023.
Returns at most {MAX_ROWS} rows; refine with LIMIT, filters, or aggregation.
"""


@tool(description=SQL_TOOL_DESCRIPTION)
def execute_readonly_sql(sql: str) -> str:
    # Failures raise rather than return: ToolErrorMiddleware turns them into an
    # error ToolMessage the model still reads, and the raise is what marks the
    # timeline row failed (a returned string is indistinguishable from a
    # successful result).
    #
    # Additional app-level safety check
    if not sql.strip().lower().startswith(
        "select"
    ) and not sql.strip().lower().startswith("with"):
        raise ToolException("Only SELECT or WITH (CTE) statements are allowed.")

    with get_ipeds_db_connection() as conn:
        cur = conn.execute(sql)
        # One extra row purely to detect (and report) truncation.
        results = cur.fetchmany(MAX_ROWS + 1)
        columns = [d[0] for d in cur.description] if cur.description else []

    if not results:
        return "Query executed successfully, but returned no rows."

    truncated = len(results) > MAX_ROWS
    results = results[:MAX_ROWS]

    # Format as a list of dictionaries for the LLM
    output = [dict(zip(columns, row)) for row in results]
    rendered = str(output)
    if truncated:
        rendered += (
            f"\n\n[Truncated: only the first {MAX_ROWS} rows are shown. "
            "Refine the query with LIMIT, filters, or aggregation.]"
        )
    return rendered


@tool
def lookup_valueset(year: int, table: str, var_name: str) -> str:
    """Return all allowed code values and their labels for a categorical IPEDS column.

    Args:
        year: The IPEDS survey start year, e.g. 2023.
        table: The data table that contains the column, e.g. "hd2023".
        var_name: The column (variable) name whose value set you want, e.g. "CONTROL".
    """
    with get_ipeds_db_connection() as con:
        # Find the valuesets table for this year (name casing varies across years).
        vs_table = _find_meta_table(con, year, "valueset")
        if vs_table is None:
            return f"No valuesets table found for year {year}."

        raw_table = table.strip()

        # codevalue is Text in every year, so the tiebreak sorts numeric-looking
        # values numerically ('10' would otherwise precede '9') and the rest as
        # text. NULLS LAST keeps missing sort keys out of the middle.
        rows = con.execute(
            f'SELECT codevalue, valuelabel FROM "{vs_table}" '
            "WHERE lower(tablename) = lower(%s) AND lower(varname) = lower(%s) "
            "ORDER BY valueorder NULLS LAST, "
            "CASE WHEN codevalue ~ '^\\s*-?[0-9]+(\\.[0-9]+)?\\s*$' "
            "THEN trim(codevalue)::numeric END NULLS LAST, "
            "codevalue NULLS LAST",
            [raw_table, var_name],
        ).fetchall()

    if not rows:
        return (
            f"No value set found for column {var_name!r} in table {raw_table!r} "
            f"(year {year}). The column may be continuous or the name may differ."
        )

    lines = [f"Value set for {raw_table}.{var_name} (year {year}):"]
    for codevalue, valuelabel in rows:
        lines.append(f"  {codevalue} — {valuelabel}")
    return "\n".join(lines)


@tool
def ask_user(question: str) -> str:
    """Ask the user to clarify abbreviations, ambiguous terms, or unclear
    requests. Use this whenever the user's objective is ambiguous and you
    cannot proceed confidently without clarification."""
    return ""


@tool
async def save_user_preference(preference: str, config: RunnableConfig) -> str:
    """Save a standing preference the user has stated about how they want
    analyses done — a default filter, a default year, a preferred output
    format. Pass one concise sentence. Use this only for a preference meant to
    outlast the current question, never for a one-off instruction."""
    # `config` is filled in by langchain and stripped from the args schema, so
    # the model only ever sees `preference`.
    user_id = (config.get("configurable") or {}).get("user_id")
    if not user_id:
        raise ToolException("No user in scope — cannot save a preference.")
    if async_session is None:
        raise ToolException("MAIN_DATABASE_URL not configured — cannot save.")

    # Its own session: the graph outlives the request-scoped one, the same
    # reason routes/sessions.py opens one to persist a turn.
    async with async_session() as db:
        items = await store.add_preference(db, uuid.UUID(user_id), preference)
    if items is None:
        raise ToolException("User not found — nothing was saved.")
    return f"Saved. The user now has {len(items)} saved preference(s)."


TOOLS = [
    get_meta_tables_by_year,
    get_table_descriptions_by_year,
    get_column_descriptions_of_table,
    execute_readonly_sql,
    lookup_valueset,
    ask_user,
    save_user_preference,
]

# The tools that pause for a human, and what the human may answer with. One
# source of truth: turn.py reads the allowed decisions off the live interrupt,
# and the eval harness strips every tool named here.
INTERRUPT_ON = {
    "ask_user": True,  # approve | edit | reject | respond
    "save_user_preference": {"allowed_decisions": ["approve", "reject"]},
}


@dynamic_prompt
def prompt_with_preferences(request: ModelRequest) -> str:
    """SYSTEM_PROMPT plus this user's saved preferences, if they have any.

    The agent is built once at startup, so the per-user part has to be applied
    per model call. turn.py puts the list in `configurable` at the top of each
    turn; reading it from there rather than the DB keeps this off the database.
    A preference approved mid-turn therefore lands on the next turn, which is
    the granularity we want anyway.
    """
    items = (get_config().get("configurable") or {}).get("preferences") or []
    if not items:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- {p}" for p in items)
    return (
        f"{SYSTEM_PROMPT}\n"
        "## User preferences\n\n"
        "Standing preferences this user has saved. Apply them unless the "
        "current message overrides them.\n\n"
        f"{lines}\n"
    )


class ToolErrorMiddleware(AgentMiddleware):
    """Turn a raising tool into an error ToolMessage instead of a dead turn.

    create_agent builds the ToolNode itself and exposes no handle_tool_errors
    kwarg, and the ToolNode default only converts ToolInvocationError (bad
    arguments) — every other exception is re-raised, escapes astream_events and
    kills the whole turn. Wrapping the call keeps the failure inside the graph:
    the model sees it and can retry, and the run still ends with on_tool_error,
    which is what turn.py marks the timeline row failed on.

    Both hooks are implemented: the app runs the async one, the eval harness
    invokes the graph synchronously.
    """

    @staticmethod
    def _error_message(request, e: Exception) -> ToolMessage:
        call = request.tool_call
        log.exception("tool %s failed", call["name"])
        return ToolMessage(
            content=f"{type(e).__name__}: {e}",
            name=call["name"],
            tool_call_id=call["id"],
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            # interrupt() travels as an exception — ask_user's pause would
            # otherwise be swallowed into an error message.
            raise
        except Exception as e:
            return self._error_message(request, e)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            return self._error_message(request, e)


class LoggingContextEditingMiddleware(ContextEditingMiddleware):
    """ContextEditingMiddleware that reports what each edit pass did.

    The edits never reach the checkpoint: the middleware applies them to a
    deepcopy it hands the model and leaves state untouched. A cleared tool
    result is therefore only observable here, around the model call itself.
    """

    async def awrap_model_call(self, request, handler):
        before = count_tokens_approximately(request.messages)

        async def logged(edited):
            results = [m for m in edited.messages if isinstance(m, ToolMessage)]
            cleared = [
                m
                for m in results
                if m.response_metadata.get("context_editing", {}).get("cleared")
            ]
            after = count_tokens_approximately(edited.messages)
            triggers = [getattr(e, "trigger", None) for e in self.edits]
            log.info(
                "context-edit: %d/%d tool results cleared, %d -> %d tokens "
                "(triggers %s)",
                len(cleared),
                len(results),
                before,
                after,
                triggers,
            )
            return await handler(edited)

        return await super().awrap_model_call(request, logged)


def build_agent(checkpointer: AsyncPostgresSaver):
    """IPEDS analysis agent with a Postgres-backed checkpointer - indexed by thread_id"""
    return create_agent(
        model=AGENT_MODEL,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            # Replaces system_prompt above on every model call; the constant
            # stays the base it builds on.
            prompt_with_preferences,
            # First of the tool-call wrappers = outermost, so it stays the
            # last-resort catch if a retrying wrapper is ever added inside it.
            ToolErrorMiddleware(),
            HumanInTheLoopMiddleware(interrupt_on=INTERRUPT_ON),
            LoggingContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=CLEAR_TOOL_USES_TRIGGER,
                        keep=4,
                        exclude_tools=["execute_readonly_sql"]
                    ),
                ],
            ),
        ],
    )
