"""Run the IPEDS agent against eval/dataset.jsonl and score it in LangSmith.

Push the dataset:      python eval/run_eval.py --push-only
See what would run:    python eval/run_eval.py --limit 10 --dry-run
Run it (costs money):  python eval/run_eval.py --limit 10

The agent under test is the production one from backend/app/agent.py - same prompt,
same tools - plus an eval-only suffix. The suffix exists because the Trend Generator's
published numbers cover a universe no question states out loud (Title IV institutions
in the 50 states and DC); without it every answer is wrong for a reason that has
nothing to do with the agent's reasoning. It is appended here rather than added to
SYSTEM_PROMPT so production behaviour is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
DATASET_NAME = "ipeds_eval"
DEFAULT_MODEL = "gpt-5.6-luna"
# Matches backend/app/config.py's RECURSION_LIMIT, which turn.py applies in production.
# LangGraph counts supersteps, and one agent loop is two of them (model, then tools),
# so 50 is roughly 24 tool-calling rounds before the run is cut off.
DEFAULT_RECURSION_LIMIT = 50

load_dotenv()

EVAL_SUFFIX = """

## Reporting universe

Unless the question says otherwise, report the standard IPEDS published universe:
Title IV institutions in the 50 states and DC. Join the year's hd<YYYY> table and
restrict to:

    pset4flg IN (1,3) AND act IN ('A','N','R','M') AND sector <> 0 AND obereg <> 9

Answer from the data available; do not ask clarifying questions.
"""


class ResponseFormat(BaseModel):
    FilteredTables: list[str] = Field(
        description="Data tables actually queried to produce the answer, with the "
        "year suffix, e.g. ['hd2022', 'ic2022_ay']. Exclude *_meta tables."
    )
    FilteredColumns: list[str] = Field(
        description="Columns used in the SELECT, WHERE, or aggregation, "
        "qualified as table.column, e.g. ['hd2022.control']."
    )
    FinalAnswer: float | None = Field(
        description="The numeric result, or null if the data cannot support one."
    )


def load_rows(args) -> list[dict]:
    rows = [json.loads(l) for l in DATASET_PATH.open(encoding="utf-8")]
    if args.latest_only:
        rows = [r for r in rows if r["latest"]]
    if args.year:
        rows = [r for r in rows if r["year"] in args.year]
    if args.variant:
        rows = [r for r in rows if r["variant"] in args.variant]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def build_agent_under_test(model_name: str, context_editing: bool):
    """The production agent, minus ask_user.

    Context editing is OFF by default: the first full run measures the model with its
    whole trajectory in view, so a wrong answer is the model's and not a consequence of
    tool output being cleared out from under it. --context-editing turns production's
    middleware back on for a comparison run.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    from backend.app.agent import (
        INTERRUPT_ON,
        SYSTEM_PROMPT,
        TOOLS,
        ToolErrorMiddleware,
    )
    from backend.app.config import CLEAR_TOOL_USES_TRIGGER

    if ":" in model_name:  # e.g. "anthropic:claude-sonnet-4-5"
        model = model_name
    else:
        import os

        os.environ.setdefault("OPENAI_API_VERSION", "2025-03-01-preview")
        from langchain_openai import AzureChatOpenAI

        model = AzureChatOpenAI(model=model_name, azure_deployment=model_name)

    # Every gated tool is dropped: with no human on the other end ask_user
    # would block forever, and save_user_preference has no user to save to.
    tools = [t for t in TOOLS if t.name not in INTERRUPT_ON]

    # Production's error handling: without it a raising tool (a bad query, a DB
    # blip) aborts the run instead of being handed back to the model to retry,
    # which would score the harness rather than the agent.
    middleware = [ToolErrorMiddleware()]
    if context_editing:
        middleware.append(
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=CLEAR_TOOL_USES_TRIGGER,
                        keep=4,
                        exclude_tools=["execute_readonly_sql"],
                    )
                ]
            )
        )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT + EVAL_SUFFIX,
        middleware=middleware,
        response_format=ResponseFormat,
    )


# --- evaluators --------------------------------------------------------------
# The Trend Generator numbers are reproducible exactly, so the bar is tight: 1%
# relative, or half a percentage point for answers that are themselves percentages.

def _tolerance_ok(pred: float, truth: float, unit: str) -> bool:
    if unit == "percent":
        return abs(pred - truth) <= 0.5
    if truth == 0:
        return abs(pred) < 1e-9
    return abs(pred - truth) / abs(truth) <= 0.01


def within_tolerance(outputs: dict, reference_outputs: dict) -> bool:
    pred = outputs.get("answer")
    if pred is None:
        return False
    return _tolerance_ok(pred, reference_outputs["answer"], reference_outputs.get("unit", "count"))


def relative_error(outputs: dict, reference_outputs: dict) -> dict:
    pred, truth = outputs.get("answer"), reference_outputs["answer"]
    if pred is None:
        return {"key": "relative_error", "score": None}
    if truth == 0:
        return {"key": "relative_error", "score": abs(pred)}
    return {"key": "relative_error", "score": abs(pred - truth) / abs(truth)}


def answered(outputs: dict) -> bool:
    """Did the agent commit to a number at all?"""
    return outputs.get("answer") is not None


def completed(outputs: dict) -> bool:
    """False when the run was cut off by the recursion limit - i.e. it looped."""
    return not outputs.get("hit_recursion_limit", False)


def tool_calls(outputs: dict) -> dict:
    """How much work the answer took. Aggregates in LangSmith, so a loop that stays
    under the limit still shows up as a fat tail next to the median."""
    return {"key": "tool_calls", "score": outputs.get("tool_calls")}


def model_turns(outputs: dict) -> dict:
    return {"key": "model_turns", "score": outputs.get("model_turns")}


def push(client, rows: list[dict], recreate: bool) -> None:
    """Upsert rows into the LangSmith dataset, keyed by the example id."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="IPEDS Trend Generator ground truth, by year and breakdown.",
        )

    existing = {
        e.metadata.get("eval_id")
        for e in client.list_examples(dataset_id=dataset.id)
        if e.metadata
    }
    new = [r for r in rows if recreate or r["id"] not in existing]
    print(f"dataset {DATASET_NAME}: {len(existing)} existing, {len(new)} to upload")
    if not new:
        return

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": r["question"]} for r in new],
        outputs=[{"answer": r["answer"], "unit": r["unit"]} for r in new],
        metadata=[
            {
                "eval_id": r["id"],
                "year": r["year"],
                "year_label": r["year_label"],
                "variant": r["variant"],
                "latest": r["latest"],
                "subject_id": r["subject_id"],
                "question_id": r["question_id"],
                "unit": r["unit"],
                "answer_type": "number",
            }
            for r in new
        ],
    )
    print(f"uploaded {len(new)} examples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--year", type=int, nargs="+")
    ap.add_argument("--variant", nargs="+", choices=["unfiltered", "filter", "breakdown"])
    ap.add_argument("--latest-only", action="store_true",
                    help="only the 45 base questions at the newest year in the DB")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT,
                    help="LangGraph supersteps before a run is cut off (default %(default)s); "
                         "a cut-off run is scored as completed=False, not an error")
    ap.add_argument("--context-editing", action="store_true",
                    help="turn production's ClearToolUsesEdit middleware back on "
                         "(off by default - the model keeps its full trajectory)")
    ap.add_argument("--experiment-prefix", default="ipeds_eval")
    ap.add_argument("--push-only", action="store_true", help="upload, run nothing")
    ap.add_argument("--recreate", action="store_true", help="re-upload every example")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selection and exit without calling the model")
    args = ap.parse_args()

    rows = load_rows(args)
    print(f"{len(rows)} examples selected")
    if args.dry_run:
        for r in rows[:20]:
            print(f"  [{r['year_label']:>8}] {r['question']}  -> {r['answer']}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
        return

    from langsmith import Client, evaluate

    client = Client()
    push(client, rows, args.recreate)
    if args.push_only:
        return

    agent = build_agent_under_test(args.model, args.context_editing)
    from langgraph.errors import GraphRecursionError

    def run_agent(inputs: dict) -> dict:
        config = {"recursion_limit": args.recursion_limit}
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": inputs["question"]}]},
                config=config,
            )
        except GraphRecursionError:
            # Recorded rather than raised: a run that loops is a result we want counted,
            # not a hole in the experiment. By definition it used the whole budget.
            return {
                "answer": None,
                "tables": [],
                "columns": [],
                "model_turns": None,
                "tool_calls": None,
                "hit_recursion_limit": True,
                "error": "GraphRecursionError",
            }

        messages = result.get("messages", [])
        sr = result.get("structured_response")
        return {
            "answer": sr.FinalAnswer if sr else None,
            "tables": sr.FilteredTables if sr else [],
            "columns": sr.FilteredColumns if sr else [],
            "model_turns": sum(1 for m in messages if getattr(m, "type", None) == "ai"),
            "tool_calls": sum(len(getattr(m, "tool_calls", None) or []) for m in messages),
            "hit_recursion_limit": False,
            "error": None if sr else "no_structured_response",
        }

    wanted = {r["id"] for r in rows}
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    examples = [
        e for e in client.list_examples(dataset_id=dataset.id)
        if e.metadata and e.metadata.get("eval_id") in wanted
    ]

    evaluate(
        run_agent,
        data=examples,
        evaluators=[within_tolerance, relative_error, answered, completed,
                    tool_calls, model_turns],
        experiment_prefix=args.experiment_prefix,
        # max_concurrency=args.concurrency,
        metadata={
            "model": args.model,
            "n": len(examples),
            "recursion_limit": args.recursion_limit,
            "context_editing": args.context_editing,
        },
    )


if __name__ == "__main__":
    main()
