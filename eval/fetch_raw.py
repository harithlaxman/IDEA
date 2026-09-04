"""Cache every Trend Generator response the eval set is built from.

Two files land in eval/raw/:

  questions.jsonl   one line per published question: {"answer": ..., "trend": ...}
  breakdowns.jsonl  one line per (question, variable): the trendTableRowOrColumn
                    response, which carries every year x every requested value code

Both are caches. build_dataset.py never touches the network; rerun this script with
--refresh when NCES publishes a new year.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TREND_HOME_URL = "https://nces.ed.gov/ipeds/trendgenerator"
ANSWER_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/answer?subjectId={}&questionId={}"
TREND_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/trend"
TREND_FILTER_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/trendTableRowOrColumn"

RAW_DIR = Path(__file__).parent / "raw"
QUESTIONS_PATH = RAW_DIR / "questions.jsonl"
BREAKDOWNS_PATH = RAW_DIR / "breakdowns.jsonl"

# Courtesy delay between requests; the API answers in well under a second.
PAUSE_S = 0.3

# Institution attributes, most-preferred first. All are columns on HD<YYYY>, so a
# question filtered by one of these tests the hd join plus a valueset lookup.
FILTER_VAR_PREFERENCE = [
    4,   # CONTROL   - Public / Private not-for-profit / Private for-profit
    59,  # HBCU      - Yes / No
    5,   # ICLEVEL   - 4-year / 2-year / less-than-2-year
    6,   # FIPS      - State
    37,  # OBEREG    - Geographic region
    1,   # sector    - control x level, so a fallback for the two above
]
MAX_FILTER_VARS = 4

# Content breakdowns split the measure itself rather than the institutions, so the
# agent has to find the right column (efytotlw, f1b02) instead of a WHERE clause.
# They are simply the expand variables that are not institution attributes - listing
# the attributes is far shorter than listing the breakdowns.
INSTITUTION_ATTR_NAMES = {
    "sector", "deggrant", "control", "control_private", "iclevel", "fips", "obereg",
    "instcat", "hdegofr1", "hbcu", "locale", "distnced",
}
MAX_EXPAND_VARS = 2

# Codes per variable. Small on purpose: every code costs an example slot later, and
# the response already spans ~20 years.
MAX_VALUE_CODES = 4
# FIPS has 50+ values; these four keep the state examples geographically spread.
PREFERRED_FIPS_CODES = ["6", "36", "48", "51"]


def fetch_questions(session: requests.Session) -> list[dict]:
    """Scrape the question index, then pull each question's answer and trend."""
    soup = BeautifulSoup(session.get(TREND_HOME_URL).content, "html.parser")
    hrefs = [a.get("href") for a in soup.select("div.qa-question a")]

    out = []
    for href in hrefs:
        sid, qid = href.split("/")[-2:]
        answer = session.get(ANSWER_URL.format(sid, qid)).json()
        time.sleep(PAUSE_S)
        trend = session.post(
            TREND_URL, json={"subjectId": sid, "questionId": qid, "filters": []}
        ).json()
        time.sleep(PAUSE_S)
        out.append({"answer": answer, "trend": trend})
        print(f"  {sid}/{qid} {answer['answer']['question']['questionText'][:60]}")
    return out


def usable_values(variable: dict) -> list[dict]:
    """Value codes worth asking about: real categories with a real label."""
    keep = []
    for v in variable.get("values") or []:
        code = str(v.get("valueCode", "")).strip()
        label = (v.get("valueLabel") or "").strip()
        if not code or not label:
            continue
        # "{Not available}" and friends are placeholders, and negative codes are
        # IPEDS missing-data markers - neither makes a sensible question.
        if label.startswith("{") or code.startswith("-"):
            continue
        keep.append({"code": code, "label": label})
    return keep


def pick_codes(var_id: int, values: list[dict]) -> list[str]:
    if var_id == 6:  # FIPS: a fixed spread of states beats the first four alphabetically
        by_code = {v["code"]: v for v in values}
        picked = [c for c in PREFERRED_FIPS_CODES if c in by_code]
        if picked:
            return picked
    return [v["code"] for v in values[:MAX_VALUE_CODES]]


def choose_variables(answer: dict) -> list[tuple[int, str]]:
    """(varId, kind) pairs to request breakdowns for, deterministically."""
    filters = {f["varId"]: f for f in answer.get("filters", [])}
    chosen: list[tuple[int, str]] = []

    for var_id in FILTER_VAR_PREFERENCE:
        if len(chosen) >= MAX_FILTER_VARS:
            break
        f = filters.get(var_id)
        if f and usable_values(f["variable"]):
            chosen.append((var_id, "filter"))

    taken = {var_id for var_id, _ in chosen}
    expands = 0
    for e in answer.get("expands", []):
        if expands >= MAX_EXPAND_VARS:
            break
        var_id = e["varId"]
        if var_id in taken:
            continue
        name = (e["variable"].get("variableName") or "").lower()
        if name in INSTITUTION_ATTR_NAMES:
            continue
        if not usable_values(e["variable"]):
            continue
        chosen.append((var_id, "expand"))
        taken.add(var_id)
        expands += 1

    return chosen


def fetch_breakdowns(session: requests.Session, questions: list[dict]) -> list[dict]:
    out = []
    for rec in questions:
        answer = rec["answer"]
        q = answer["answer"]["question"]
        sid, qid = q["subjectId"], q["questionId"]
        by_id = {v["varId"]: v for v in answer.get("filters", []) + answer.get("expands", [])}

        for var_id, kind in choose_variables(answer):
            variable = by_id[var_id]["variable"]
            values = usable_values(variable)
            codes = pick_codes(var_id, values)
            if not codes:
                continue
            resp = session.post(
                TREND_FILTER_URL,
                json={
                    "subjectId": sid,
                    "questionId": qid,
                    "columnVarId": var_id,
                    "columnVarValueCodes": codes,
                },
            ).json()
            time.sleep(PAUSE_S)
            out.append(
                {
                    "subject_id": sid,
                    "question_id": qid,
                    "var_id": var_id,
                    "var_name": variable.get("variableName"),
                    "var_title": variable.get("variableTitle"),
                    "kind": kind,
                    "requested_codes": codes,
                    "labels": {v["code"]: v["label"] for v in values},
                    "response": resp,
                }
            )
            print(f"  {sid}/{qid} {kind} {variable.get('variableName')} {codes}")
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="refetch even if cached")
    args = ap.parse_args()

    session = requests.Session()

    if QUESTIONS_PATH.exists() and not args.refresh:
        questions = [json.loads(l) for l in QUESTIONS_PATH.open(encoding="utf-8")]
        print(f"questions.jsonl: {len(questions)} cached (--refresh to refetch)")
    else:
        print("fetching questions...")
        questions = fetch_questions(session)
        write_jsonl(QUESTIONS_PATH, questions)
        print(f"questions.jsonl: {len(questions)} written")

    if BREAKDOWNS_PATH.exists() and not args.refresh:
        n = sum(1 for _ in BREAKDOWNS_PATH.open(encoding="utf-8"))
        print(f"breakdowns.jsonl: {n} cached (--refresh to refetch)")
        return

    print("fetching breakdowns...")
    breakdowns = fetch_breakdowns(session, questions)
    write_jsonl(BREAKDOWNS_PATH, breakdowns)
    print(f"breakdowns.jsonl: {len(breakdowns)} written")


if __name__ == "__main__":
    main()
