"""Rewrite each templated question as something a person would actually type.

One rewrite per example - never several variants of the same item. Examples are sent
ten at a time purely to save round trips.

The rewrite must not change what is being asked, so every returned string is checked
for the year and for the filter/breakdown dimension before it is accepted. Anything
that fails falls back to its template, which is always kept in the record.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
os.environ.setdefault("OPENAI_API_VERSION", "2025-03-01-preview")

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
DEFAULT_MODEL = "gpt-5.6-luna"
BATCH_SIZE = 10

SYSTEM = """You rewrite templated IPEDS data questions into natural questions, of the
kind a higher-education analyst would type into a chat box.

Rules for every rewrite:
- Ask for exactly the same measure. Do not generalise, narrow, or editorialise.
- Keep the year. Use the year label given, or an equally unambiguous form of it.
- If the item has a dimension (a filter or a breakdown), the rewrite must still make
  that dimension explicit - say HBCUs, or women, or public institutions, in plain
  English rather than as a parenthetical code.
- Add no condition that is not in the template.
- One sentence. Vary the phrasing across items; do not use the same opening twice in
  a batch.
- Plain text, no markdown, no quotes around the question.

Return one rewrite per input id."""

# Labels too generic to look for on their own: "Yes" tells you nothing, so the check
# looks for the variable instead.
GENERIC_LABELS = {"yes", "no", "total"}

# Natural phrasings the check should accept for a value label.
SYNONYMS = {
    "male": ["male", "men"],
    "female": ["female", "women"],
    "yes": ["yes"],
    "hbcu": ["hbcu", "historically black"],
    "public": ["public"],
    "private not-for-profit": ["not-for-profit", "nonprofit", "non-profit"],
    "private for-profit": ["for-profit"],
    "four or more years": ["4-year", "four-year", "four or more"],
    "at least 2 but less than 4 years": ["2-year", "two-year", "at least 2"],
    "less than 2 years (below associate)": ["less-than-2-year", "less than 2"],
}

STOPWORDS = {"the", "and", "or", "of", "in", "at", "a", "an", "than", "less", "more"}


class Rewrite(BaseModel):
    id: str = Field(description="The id of the input item, copied verbatim.")
    question: str = Field(description="The rewritten question.")


class Rewrites(BaseModel):
    rewrites: list[Rewrite]


def year_ok(text: str, label: str) -> bool:
    text = text.lower()
    if label.lower() in text:
        return True
    years = re.findall(r"\d{4}", label)
    if not years:
        return False
    start = int(years[0])
    forms = [str(start), str(start + 1), f"{start}-{str(start + 1)[2:]}", f"{start}-{start + 1}"]
    return any(f in text for f in forms)


def dimension_ok(text: str, dimension: dict | None) -> bool:
    if dimension is None:
        return True
    text = text.lower()
    label = (dimension["label"] or "").lower()
    var_name = (dimension["var_name"] or "").lower()

    for key in (label, var_name):
        for phrase in SYNONYMS.get(key, []):
            if phrase in text:
                return True

    if label in GENERIC_LABELS:
        # "Yes"/"No" only mean something next to the variable, e.g. HBCU.
        return var_name in text or (dimension["var_title"] or "").lower() in text

    if label in text:
        return True

    # Otherwise accept if most of the label's real words survived the rewrite.
    tokens = [t for t in re.findall(r"[a-z0-9-]{3,}", label) if t not in STOPWORDS]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in text)
    return hits >= max(1, len(tokens) // 2)


def validate(rewrite: str, example: dict) -> str | None:
    """The accepted rewrite, or None with the reason printed by the caller."""
    text = rewrite.strip()
    if not text or len(text) > 2.5 * len(example["question_template"]):
        return None
    if not year_ok(text, example["year_label"]):
        return None
    if not dimension_ok(text, example["dimension"]):
        return None
    return text


def render_item(example: dict) -> dict:
    item = {
        "id": example["id"],
        "template": example["question_template"],
        "year_label": example["year_label"],
        "measure": example["metric"],
    }
    if example["dimension"]:
        item["dimension"] = f"{example['dimension']['var_title']}: {example['dimension']['label']}"
    return item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print rewrites, write nothing")
    ap.add_argument("--limit", type=int, help="only process the first N examples")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true",
                    help="rewrite examples that already differ from their template")
    args = ap.parse_args()

    from langchain_openai import AzureChatOpenAI

    rows = [json.loads(l) for l in DATASET_PATH.open(encoding="utf-8")]
    todo = [r for r in rows
            if args.force or r["question"] == r["question_template"]]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} of {len(rows)} examples to rewrite")

    model = AzureChatOpenAI(model=args.model, azure_deployment=args.model)
    structured = model.with_structured_output(Rewrites)
    by_id = {r["id"]: r for r in rows}

    accepted = rejected = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        payload = json.dumps([render_item(e) for e in batch], indent=1)
        result = structured.invoke(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": payload}]
        )
        returned = {r.id: r.question for r in result.rewrites}

        for example in batch:
            raw = returned.get(example["id"], "")
            good = validate(raw, example)
            if good is None:
                rejected += 1
                print(f"  [fallback] {example['id']}: {raw[:90]!r}")
                continue
            accepted += 1
            by_id[example["id"]]["question"] = good
            if args.dry_run:
                print(f"\n  {example['id']}")
                print(f"    template: {example['question_template']}")
                print(f"    rewrite : {good}")
        print(f"batch {start // args.batch_size + 1}: "
              f"{accepted} accepted, {rejected} fell back to template")

    if args.dry_run:
        print("\ndry run - dataset.jsonl unchanged")
        return

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {DATASET_PATH} ({accepted} rewritten, {rejected} kept as template)")


if __name__ == "__main__":
    main()
