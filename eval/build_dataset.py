"""Turn the cached Trend Generator responses into eval examples.

Reads eval/raw/{questions,breakdowns}.jsonl, writes eval/dataset.jsonl. No network.

Two things drive every decision here:

* A trend row's *number* is the DB table's year suffix (row 2023 -> hd2023, effy2023,
  f2223_f1a); the row's *label* is the human data year and differs by survey (HD 2023 is
  "2023-24", EFFY 2023 is "2022-23"). Questions get the label, sampling gets the number.
* Our database stops at 2023: there is no hd2024 and no f2324_*. Rows outside
  2004..2023 are dropped, however good the ground truth is.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
QUESTIONS_PATH = RAW_DIR / "questions.jsonl"
BREAKDOWNS_PATH = RAW_DIR / "breakdowns.jsonl"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

MIN_YEAR, MAX_YEAR = 2004, 2023  # inclusive DB coverage
TARGET_SIZE = 400
SEED = 0

# Roughly how the sampled remainder splits. The latest-year block is all unfiltered,
# so the finished set leans a little further toward unfiltered than this.
VARIANT_MIX = {"unfiltered": 0.4, "filter": 0.3, "breakdown": 0.3}


def parse_value(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    if not text or text in {".", "-", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def unit_of(question: dict) -> str:
    """What the number means, so the scorer can pick a sensible tolerance."""
    text = f"{question['questionText']} {question.get('reportText', '')}".lower()
    if "percent" in text or "rate" in text:
        return "percent"
    if "in thousands" in text:
        return "thousands_of_dollars"
    if any(k in text for k in ("average amount", "salary", "tuition", "expenses per")):
        return "dollars"
    return "count"


# What a year label actually counts, from the question's own dataYearLabel. A
# graduation-rate row labelled 2017 is the 2017 *cohort*, reported in gr2023 - saying
# only "for 2017" would ask a different question from the one we have an answer to.
YEAR_PHRASE = {
    "Academic Year": "In academic year {}",
    "Fall": "In fall {}",
    "Fiscal year": "In fiscal year {}",
    "Cohort Year": "For the {} cohort",
    "Completion Year": "In completion year {}",
    "Aid year": "In aid year {}",
    "Year": "In {}",
}


def year_phrase(year_label: str, year_kind: str | None) -> str:
    return YEAR_PHRASE.get(year_kind or "", "In {}").format(year_label)


def template(question_text: str, year_label: str, year_kind: str | None,
             dimension: dict | None) -> str:
    body = question_text[0].lower() + question_text[1:] if question_text else ""
    out = f"{year_phrase(year_label, year_kind)}, {body}"
    if dimension:
        out += f" ({dimension['var_title']}: {dimension['label']})"
    return out


def make_example(q: dict, year: int, year_label: str, value: float,
                 dimension: dict | None, variant: str) -> dict:
    qy = q["questionYear"]
    year_kind = q.get("dataYearLabel")
    slug = "base" if dimension is None else f"{dimension['var_name']}{dimension['code']}".lower()
    return {
        "id": f"s{q['subjectId']}-q{q['questionId']}-y{year}-{slug}",
        "subject_id": q["subjectId"],
        "question_id": q["questionId"],
        "year": year,
        "year_label": year_label,
        "year_kind": year_kind,
        "variant": variant,
        "latest": False,
        "dimension": dimension,
        "question_template": template(q["questionText"], year_label, year_kind, dimension),
        "question": template(q["questionText"], year_label, year_kind, dimension),
        "answer": value,
        "unit": unit_of(q),
        "metric": q.get("reportText") or q["questionText"],
        "source": {
            "table_latest": qy.get("tableName"),
            "select_list": qy.get("selectList"),
            "condition_clause": qy.get("conditionClause"),
        },
    }


def build_pool() -> tuple[list[dict], dict[tuple[int, int], dict]]:
    """All candidate examples, plus the question record for each (subject, question)."""
    pool: list[dict] = []
    questions: dict[tuple[int, int], dict] = {}

    for line in QUESTIONS_PATH.open(encoding="utf-8"):
        rec = json.loads(line)
        q = rec["answer"]["answer"]["question"]
        questions[(q["subjectId"], q["questionId"])] = q
        for row in rec["trend"]["trend"]["rows"]:
            year = int(row["year"])
            if not MIN_YEAR <= year <= MAX_YEAR:
                continue
            cols = row.get("columnValues") or []
            if len(cols) < 2:
                continue
            value = parse_value(cols[1])
            if value is None:
                continue
            label = row.get("yearLabel") or cols[0]
            pool.append(make_example(q, year, str(label), value, None, "unfiltered"))

    for line in BREAKDOWNS_PATH.open(encoding="utf-8"):
        bd = json.loads(line)
        q = questions[(bd["subject_id"], bd["question_id"])]
        variant = "filter" if bd["kind"] == "filter" else "breakdown"
        for row in bd["response"].get("rows") or []:
            year = int(row["yearNumber"])
            if not MIN_YEAR <= year <= MAX_YEAR:
                continue
            for col in row.get("columnValues") or []:
                code = str(col.get("valueCode") or "")
                # The "" column is a total - and in a breakdown response it totals only
                # the codes we asked for, so it is never usable ground truth.
                if not code or code not in bd["requested_codes"]:
                    continue
                value = parse_value(col.get("value"))
                if value is None:
                    continue
                dimension = {
                    "var_id": bd["var_id"],
                    "var_name": bd["var_name"],
                    "var_title": bd["var_title"],
                    "code": code,
                    "label": bd["labels"].get(code, code),
                }
                pool.append(
                    make_example(q, year, str(row["year"]), value, dimension, variant)
                )

    return pool, questions


def latest_year_block(pool: list[dict]) -> list[dict]:
    """One unfiltered example per question, at the newest year our DB holds.

    Usually labelled 2023-24; the EFFY-backed questions land on 2022-23, because
    their 2023-24 figures live in effy2024 and we do not have that table.
    """
    best: dict[tuple[int, int], dict] = {}
    for ex in pool:
        if ex["variant"] != "unfiltered":
            continue
        key = (ex["subject_id"], ex["question_id"])
        if key not in best or ex["year"] > best[key]["year"]:
            best[key] = ex
    out = []
    for ex in best.values():
        ex = dict(ex, latest=True)
        out.append(ex)
    return sorted(out, key=lambda e: (e["subject_id"], e["question_id"]))


def sample(pool: list[dict], chosen: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Fill the remaining slots, balanced across years, variants, and questions."""
    taken_ids = {e["id"] for e in chosen}
    # A question may appear twice in one year as long as the two ask different things
    # (the plain figure, and a figure split by some dimension). Without this the
    # latest-year block would lock every question out of its own newest year.
    used = {(e["subject_id"], e["question_id"], e["year"], e["variant"]) for e in chosen}
    per_question = Counter((e["subject_id"], e["question_id"]) for e in chosen)

    by_year: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for ex in pool:
        if ex["id"] in taken_ids:
            continue
        by_year[ex["year"]][ex["variant"]].append(ex)
    for variants in by_year.values():
        for items in variants.values():
            rng.shuffle(items)

    years = list(range(MIN_YEAR, MAX_YEAR + 1))
    remaining = target - len(chosen)
    quota = {y: remaining // len(years) for y in years}
    for y in years[: remaining % len(years)]:
        quota[y] += 1

    picked: list[dict] = []
    # Two passes: the first honours each year's quota, the second spends whatever
    # thin early years could not absorb.
    for _ in range(2):
        for year in years:
            want = quota[year]
            if want <= 0:
                continue
            order = []
            for variant, share in VARIANT_MIX.items():
                order += [variant] * max(1, round(share * want))
            rng.shuffle(order)

            for variant in order:
                if quota[year] <= 0 or len(picked) >= remaining:
                    break
                candidates = by_year[year][variant] or [
                    c for v in by_year[year].values() for c in v
                ]
                # Spread across questions: always take from the question used least so far.
                best_i = None
                best_key = None
                for i, cand in enumerate(candidates):
                    key = (cand["subject_id"], cand["question_id"])
                    if (*key, year, variant) in used:
                        continue
                    rank = per_question[key]
                    if best_key is None or rank < best_key:
                        best_i, best_key = i, rank
                    if rank == 0:
                        break
                if best_i is None:
                    continue
                cand = candidates.pop(best_i)
                key = (cand["subject_id"], cand["question_id"])
                picked.append(cand)
                per_question[key] += 1
                used.add((*key, year, cand["variant"]))
                quota[year] -= 1
                if len(picked) >= remaining:
                    return picked

        leftover = sum(q for q in quota.values() if q > 0)
        if leftover == 0:
            break
        # Push the shortfall onto years that still have candidates.
        spare = [y for y in years if any(by_year[y].values())]
        if not spare:
            break
        for i in range(leftover):
            quota[spare[i % len(spare)]] += 1
        for y in years:
            if not any(by_year[y].values()):
                quota[y] = 0

    return picked


def summarise(rows: list[dict]) -> None:
    per_year = Counter(r["year"] for r in rows)
    per_variant = Counter(r["variant"] for r in rows)
    per_question = Counter((r["subject_id"], r["question_id"]) for r in rows)
    per_subject = Counter(r["subject_id"] for r in rows)

    print(f"\n{len(rows)} examples")
    print(f"variants: {dict(per_variant)}")
    print(f"latest-year block: {sum(1 for r in rows if r['latest'])}")
    print(f"questions covered: {len(per_question)} (min {min(per_question.values())} "
          f"examples, max {max(per_question.values())})")
    print(f"subjects: {dict(sorted(per_subject.items()))}")
    print("\nyear  n   variants")
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        rows_y = [r for r in rows if r["year"] == year]
        mix = Counter(r["variant"] for r in rows_y)
        print(f"{year}  {len(rows_y):3d}  " + " ".join(f"{k}={v}" for k, v in sorted(mix.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=TARGET_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool, _ = build_pool()
    print(f"pool: {len(pool)} candidate examples "
          f"({Counter(e['variant'] for e in pool)})")

    chosen = latest_year_block(pool)
    chosen += sample(pool, chosen, args.size, rng)
    chosen.sort(key=lambda e: (e["year"], e["subject_id"], e["question_id"], e["id"]))

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row) + "\n")

    summarise(chosen)
    print(f"\nwrote {DATASET_PATH}")


if __name__ == "__main__":
    main()
