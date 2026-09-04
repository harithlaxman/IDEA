"""What one turn *does*: drives the agent built in agent.py, normalizing
LangGraph events into the timeline the frontend renders.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from .config import CONTEXT_WINDOW, RECURSION_LIMIT

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def _delta_text(chunk) -> str:
    """Plain text from one streamed chunk. Anthropic content may be a list of
    blocks; non-text ones (tool-call args, thinking) contribute nothing."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out)


def get_thread_config(
    thread_id: str,
    user_id: str | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    """The LangGraph config for one session's checkpoint thread.

    ``preferences`` is read back by agent.prompt_with_preferences on every model
    call of this turn — passed through config so the graph never touches the DB
    to build a prompt.
    """
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if user_id is not None:
        configurable["user_id"] = user_id
    if preferences:
        configurable["preferences"] = preferences
    return {"configurable": configurable, "recursion_limit": RECURSION_LIMIT}


@dataclass
class Pending:
    """A thread paused on a human-in-the-loop tool call.

    ``allowed`` is the middleware's own ``allowed_decisions`` for the paused
    tool, so callers can branch on what the human may answer with rather than
    on which tool it happens to be.
    """

    interrupt_id: str
    name: str
    args: dict = field(default_factory=dict)
    allowed: list[str] = field(default_factory=list)
    count: int = 1

    @property
    def question(self) -> str:
        return self.args.get("question", "")

    @property
    def preference(self) -> str:
        return self.args.get("preference", "")

    @property
    def is_clarify(self) -> bool:
        """True when the human answers with text, false when with a decision."""
        return "respond" in self.allowed


async def pending_interrupt(agent, config) -> Pending | None:
    """The interrupt this thread is paused on, or None if it is running free."""
    state = await agent.aget_state(config)
    for interrupt in state.interrupts:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        requests = value.get("action_requests") or []
        if not requests:
            continue
        configs = value.get("review_configs") or []
        allowed = configs[0].get("allowed_decisions", []) if configs else []
        return Pending(
            interrupt_id=interrupt.id,
            name=requests[0].get("name", ""),
            args=requests[0].get("args") or {},
            allowed=list(allowed),
            # One decision per gated call is required on resume; the middleware
            # raises if the counts disagree.
            count=len(requests),
        )
    return None


# --- Context meter -----------------------------------------------------------
#
# Anthropic reports usage once per model call. langchain_anthropic folds
# cache_read/cache_creation back into usage_metadata["input_tokens"], so that
# field is the true, cache-inclusive size of everything the call sent — system
# prompt, tool schemas, every prior message. Adding output_tokens gives what the
# next call starts from. Anything appended since (tool results, a user message)
# has no provider count yet, so it is approximated with the same counter
# ContextEditingMiddleware uses to decide when to clear tool results.


def _usage_from_messages(messages) -> tuple[int, bool]:
    """``(tokens, exact)`` for a message list. ``exact`` is False if estimated."""
    for i in range(len(messages) - 1, -1, -1):
        usage = getattr(messages[i], "usage_metadata", None)
        if isinstance(messages[i], AIMessage) and usage:
            counted = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            rest = messages[i + 1 :]
            return counted + _approx(rest), not rest
    # No model call on this thread yet — every message is an estimate.
    return _approx(messages), not messages


def _approx(messages) -> int:
    """Approximate tokens for messages the provider has not counted."""
    if not messages:
        return 0
    try:
        return count_tokens_approximately(messages)
    except Exception:
        # The meter is cosmetic; never let it break a turn.
        return 0


def _usage_payload(tokens: int, exact: bool) -> dict:
    return {"tokens": max(tokens, 0), "window": CONTEXT_WINDOW, "exact": exact}


def _as_message(value) -> BaseMessage:
    """Coerce a tool result to something count_tokens_approximately accepts."""
    if isinstance(value, BaseMessage):
        return value
    return ToolMessage(content=str(value), tool_call_id="")


async def context_usage(agent, config) -> dict:
    """What the thread's context currently carries, read from the checkpoint."""
    try:
        state = await agent.aget_state(config)
        tokens, exact = _usage_from_messages(state.values.get("messages") or [])
    except Exception:
        return _usage_payload(0, True)
    return _usage_payload(tokens, exact)


async def repair_orphan_tool_calls(agent, config) -> bool:
    """Answer tool calls a died turn left stranded. True if it repaired any."""
    state = await agent.aget_state(config)
    messages = state.values.get("messages") or []
    if not messages:
        return False
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    # The tail isn't enough: create_agent fans a batch out as one Send per tool
    # call, so a turn cancelled mid-batch leaves the finished results committed
    # and the tail a ToolMessage, with the stranded ids an AIMessage or two back.
    # Only the newest such AIMessage can hold orphans — the graph never routes
    # back to the model with earlier calls outstanding.
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    orphans = [
        tc for tc in (last_ai.tool_calls if last_ai else []) if tc["id"] not in answered
    ]
    if not orphans:
        return False
    log.warning(
        "Repairing %d orphaned tool call(s) on thread %s",
        len(orphans),
        config["configurable"]["thread_id"],
    )
    await agent.aupdate_state(
        config,
        {
            "messages": [
                ToolMessage(
                    content="Cancelled — the turn ended before this ran.",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in orphans
            ]
        },
    )
    return True


async def process_query(
    query: str,
    thread_id: str,
    user_id: str,
    agent,
    *,
    preferences: list[str] | None = None,
    decision: str | None = None,
):
    """Drive the agent for one user turn, streaming its timeline as it happens.

    Start vs. resume is decided here from the checkpoint, never by the browser:
    a paused thread takes the answer as a Command(resume=…) keyed by interrupt
    id, so a stale tab or a mid-turn reload is harmless and a resumed turn
    renders like any other. What *shape* that answer takes still comes from the
    checkpoint too — the paused tool's allowed decisions say whether the human
    replies with text (``query``) or with a verdict (``decision``).

    The turn is a *timeline* — an ordered list of what the agent said and did,
    one entry per contiguous run of prose and one per tool call::

        [{"kind": "text", "text": "I'll find the right table first."},
         {"kind": "tool", "name": "get_table_descriptions_by_year", "ok": True},
         {"kind": "text", "text": "Texas gained the most, at +41,200 …"}]

    On a successful turn a trailing text entry is the answer: renderers show it
    as the body and leave it out of the log. A failed turn promoted nothing, so
    its prose stays in the log and the error becomes the body. app.js and
    Message.log_entries apply the same rule. Status is separate — transient
    chatter that disappears once the turn ends.

    Yields:
        {"type": "status", "text": str}
        {"type": "usage",  "tokens": int, "window": int, "exact": bool}
        {"type": "text",   "seq": int, "delta": str}
        {"type": "tool",   "phase": "start"|"end", "name": str, "run_id": str,
                           "ok": bool (end only)}
        {"type": "clarify", "question": str, "timeline": list}
        {"type": "confirm", "preference": str, "timeline": list}
        {"type": "final",  "text": str, "timeline": list, "artifacts": list,
                           "error": str | None}

    A turn ends on exactly one of clarify, confirm, or final. clarify and
    confirm are the two pause shapes: clarify wants text back, confirm wants
    approve/reject. Both already end their timeline with what they are asking,
    so both persist just like a final's. ``seq`` is the delta's index into the
    timeline.

    ``usage`` is transient like ``status`` — the context meter, not part of the
    persisted turn. It is exact straight after a model call and estimated in
    between, so it climbs as each tool result lands rather than freezing until
    the answer does.
    """
    config = get_thread_config(thread_id, user_id, preferences)

    pending = await pending_interrupt(agent, config)
    if pending:
        # Branch on what the paused tool accepts, not on its name: a text
        # answer resumes ask_user, a verdict resumes an approval gate. One
        # decision per gated call, or the middleware rejects the batch.
        if pending.is_clarify:
            d: dict[str, Any] = {"type": "respond", "message": query}
        elif decision == "approve":
            d = {"type": "approve"}
        else:
            d = {"type": "reject", "message": "User declined. Do not save it."}
        graph_input = Command(
            resume={pending.interrupt_id: {"decisions": [d] * pending.count}}
        )
    else:
        await repair_orphan_tool_calls(agent, config)
        graph_input = {"messages": [{"role": "user", "content": query}]}

    # Seeded from the checkpoint, then kept current by the stream: exact after
    # every model call, estimated after every tool result.
    usage = await context_usage(agent, config)
    tokens = usage["tokens"] + _approx([{"role": "user", "content": query}])
    yield {"type": "usage", **_usage_payload(tokens, False)}

    error = None
    timeline: list[dict] = []
    # Index of the text entry deltas append to; None when the next delta should
    # open a new one (before the first token, and after anything that closes a
    # segment).
    cur_text: int | None = None
    # run_id → timeline index, so on_tool_end updates the entry its own
    # on_tool_start appended (the same tool often runs several times a turn).
    pending_tools: dict[str, int] = {}
    try:
        # durability="sync" (not the default "async"): a client disconnect
        # cancels the graph mid-step, and syncing each checkpoint first is what
        # leaves a coherent one behind for the next turn to pick up.
        async for ev in agent.astream_events(
            graph_input, config, version="v2", durability="sync"
        ):
            kind = ev["event"]
            if kind == "on_chat_model_start":
                cur_text = None
                yield {"type": "status", "text": "thinking…"}
            elif kind == "on_chat_model_end":
                model_usage = getattr(ev["data"].get("output"), "usage_metadata", None)
                if model_usage:
                    tokens = model_usage.get("input_tokens", 0) + model_usage.get(
                        "output_tokens", 0
                    )
                    yield {"type": "usage", **_usage_payload(tokens, True)}
            elif kind == "on_tool_start":
                cur_text = None
                run_id = str(ev.get("run_id", ""))
                pending_tools[run_id] = len(timeline)
                timeline.append({"kind": "tool", "name": ev["name"], "ok": True})
                yield {
                    "type": "tool",
                    "phase": "start",
                    "name": ev["name"],
                    "run_id": run_id,
                }
            elif kind == "on_tool_end":
                run_id = str(ev.get("run_id", ""))
                output = ev["data"].get("output")
                # on_tool_end.data is only {"output", "input"} — a raising tool
                # never reaches here at all (see on_tool_error below). What does
                # arrive marked failed is an error ToolMessage returned through
                # the normal path, e.g. ToolNode's unknown-tool reply.
                ok = getattr(output, "status", None) != "error"
                idx = pending_tools.pop(run_id, None)
                if idx is not None:
                    timeline[idx]["ok"] = ok
                yield {
                    "type": "tool",
                    "phase": "end",
                    "name": ev["name"],
                    "run_id": run_id,
                    "ok": ok,
                }
                # No provider count exists for a result the model has yet to
                # see, so estimate it; the next model call replaces the whole
                # figure with an exact one.
                tokens += _approx([_as_message(output)])
                yield {"type": "usage", **_usage_payload(tokens, False)}
            elif kind == "on_tool_error":
                # A tool that raises ends its run here and never emits
                # on_tool_end, so this is the only place its row can be closed —
                # without it the frontend spinner runs forever, on a turn that
                # ToolErrorMiddleware usually lets the agent recover from.
                #
                # Deliberately does not set `error`: the agent sees the failure
                # as a tool result and generally retries, and failing the turn
                # here would discard an answer it went on to produce. A fatal
                # exception still propagates to the except below.
                run_id = str(ev.get("run_id", ""))
                idx = pending_tools.pop(run_id, None)
                if idx is not None:
                    timeline[idx]["ok"] = False
                yield {
                    "type": "tool",
                    "phase": "end",
                    "name": ev["name"],
                    "run_id": run_id,
                    "ok": False,
                }
                # The middleware's error message lands in state like any other
                # tool result, so it counts against the window the same way.
                tokens += _approx([_as_message(str(ev["data"].get("error")))])
                yield {"type": "usage", **_usage_payload(tokens, False)}
            elif kind == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                if not chunk:
                    continue
                delta = _delta_text(chunk)
                # Tool-args/thinking chunks yield "" — emitting them would open
                # an empty text entry that renders as a blank row.
                if not delta:
                    continue
                if cur_text is None:
                    timeline.append({"kind": "text", "text": ""})
                    cur_text = len(timeline) - 1
                timeline[cur_text]["text"] += delta
                yield {"type": "text", "seq": cur_text, "delta": delta}
    except Exception as e:
        error = str(e)

    # Outside the try on purpose: a turn can pause on a gated tool and *then*
    # have the stream fail, and a paused turn that never says so strands the
    # thread. So a discovered pause wins over reporting the error. Runs on
    # resumed turns too — the agent may pause a second time.
    pending = await pending_interrupt(agent, config)
    if pending:
        # What the agent is waiting on is this turn's answer, so it goes on as
        # the trailing text entry — the rule every other turn follows.
        if pending.is_clarify:
            timeline.append({"kind": "text", "text": pending.question})
            yield {
                "type": "clarify",
                "question": pending.question,
                "timeline": timeline,
            }
        else:
            prompt = f'Save this preference?\n\n"{pending.preference}"'
            timeline.append({"kind": "text", "text": prompt})
            yield {
                "type": "confirm",
                "preference": pending.preference,
                "timeline": timeline,
            }
        return

    ends_on_text = bool(timeline) and timeline[-1]["kind"] == "text"
    if not error and not ends_on_text:
        # Graph stopped without a final reply (recursion limit, or cancelled
        # after a tool result). Reported, not persisted as an empty answer.
        error = "The agent stopped before answering."
    # A failed turn has no answer: its prose stays in the timeline as log
    # content and the error renders as the body instead.
    answer = timeline[-1]["text"] if (ends_on_text and not error) else ""
    yield {
        "type": "final",
        "text": answer,
        "timeline": timeline,
        "artifacts": [],
        "error": error,
    }
