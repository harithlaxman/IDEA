"""Post-process the ingested Postgres DB: LLM-summarize table descriptions.

Runs *after* `ingest.py`. Works one (year, survey) group at a time rather than
one table at a time, because the redundancy this removes is redundancy *within a
survey*: all six Fall Enrollment tables repeat the same paragraph about semester
calendars and the October 15 reporting date, and only a clause or two actually
tells them apart.

Each group produces two things:

  * a survey summary, holding what the group has in common, written to a new
    `surveys<YY>_meta` table; and
  * a per-table summary, holding only what separates that table from its
    siblings, written to `tables<YY>_meta.description_summary` (the column name
    the agent's `get_table_descriptions_by_year` already coalesces to).

Resumable: a group whose survey row exists and whose tables are all summarized
is skipped, so an interrupted run picks up where it stopped.

Every model call is also banked to CSV under `summaries/` (see --csv-dir), so a
database rebuilt by `ingest.py` can be re-summarized with --from-csv, which reads
those files and writes them straight to Postgres without calling the model.

Config (from environment / a local .env):
    IPEDS_DATABASE_URL            Postgres URL; DATABASE_URL is accepted as a fallback.
    AZURE_OPENAI_ENDPOINT         e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY          Azure OpenAI API key; AZURE_OPENAI_KEY also accepted.
    AZURE_OPENAI_DEPLOYMENT_NAME  Deployment (model) name to call.
    AZURE_OPENAI_API_VERSION      Optional; defaults to a recent stable version.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from psycopg import sql
from tqdm import tqdm

# Load configuration (IPEDS_DATABASE_URL, Azure creds, ...) from a local .env if present.
load_dotenv()

# The rest of the repo reads IPEDS_DATABASE_URL; DATABASE_URL is what earlier
# versions of this script used, so it stays accepted rather than silently failing.
DATABASE_URL = "postgresql://postgres:idea@localhost:5432/idea_ipeds"
# os.environ.get("IPEDS_DATABASE_URL") or os.environ.get("DATABASE_URL")

# Structured Outputs (response_format=json_schema) needs 2024-08-01-preview or
# later; this GA version qualifies.
DEFAULT_API_VERSION = "2024-10-21"

# Matches the metadata catalogs this script reads, e.g. 'tables23_meta'.
TABLES_META_RE = re.compile(r"^tables(\d{2})_meta$", re.IGNORECASE)

# Excel writes carriage returns into the workbook as this literal escape.
_CR_ESCAPE_RE = re.compile(r"_x000[Dd]_")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Every model call is banked here as CSV, so a re-ingested database can be
# re-summarized with --from-csv instead of paying for the tokens again.
DEFAULT_CSV_DIR = Path(__file__).resolve().parent.parent / "data" / "summaries"
SURVEYS_CSV_FIELDS = ["survey", "table_count", "summary"]
TABLES_CSV_FIELDS = ["survey", "tablename", "description_summary"]

SYSTEM_PROMPT = """\
You write navigation metadata for an AI agent that explores the IPEDS database by \
choosing which table to query.

You are given every table belonging to ONE IPEDS survey component for ONE year: its \
name, title, year coverage, and full description. Those descriptions were written for \
a human reading a codebook, so they repeat the same survey-wide boilerplate in every \
single table. Your job is to say that shared material once and then say only what \
tells the tables apart.

Produce two things.

1. survey_summary — what ALL of these tables share: the collection window and \
reporting rules, the unit of observation, the universe and eligibility, and caveats \
that apply across the whole survey. Write it so a reader can decide whether this \
survey is worth exploring at all. 2-4 sentences.

2. tables — one summary per table. Each summary must stand on its own: the agent \
often reads it without the survey summary beside it, so open by naming what the table \
actually holds in plain subject terms ("Fall enrollment counts broken down by ...", \
"Completions awarded by ...") rather than with a bare fragment ("Broken down by ..."). \
After that opening phrase, carry ONLY what separates the table from its siblings AND \
would change which table an agent decides to query.

In each table summary, KEEP:
- the breakdown dimensions that distinguish it from its siblings (race/ethnicity vs. \
age vs. distance education vs. residence, and so on)
- the variable(s) that define row grain or make a record unique — name them explicitly
- population restrictions that decide whether the table can answer a question at all: \
undergraduate-only, degree-granting only, non-open-admission only, 4-year vs. \
less-than-4-year, separate GASB/FASB/for-profit forms
- availability constraints: columns that exist only from a given year forward, and \
collection that happens only in even- or odd-numbered years. For the even/odd rule, \
give the rule as the catalog states it and then resolve it against the year in the \
prompt, whose parity is labelled for you -- "reported only in odd-numbered years, so \
2004 (even) has no data" or "... so 2023 (odd) has data". Never turn the rule into a \
bare claim about "this year" without checking that parity against the label: getting \
it backwards inverts whether the data exists at all, which is worse than omitting it

CUT:
- anything you put in survey_summary: collection windows, reporting dates, calendar \
rules, the survey's universe. Naming the survey's subject in the opening phrase is \
not a repeat; restating its rules is
- the table's own title and year coverage restated wholesale. Both are stored \
separately and displayed next to your summary, so echoing every qualifier wastes the \
reader's attention -- the short subject phrase that makes the summary readable on its \
own is the only overlap worth keeping
- response rates and data-quality commentary. "Approximately only about 50 percent of \
institutions respond in optional years" is OUT. The adjacent "optional for \
institutions to report in even-numbered years" STAYS, because it tells the agent \
whether the year it is querying has data at all
- definitional prose: what a term means conceptually, how a ratio is computed. The \
agent needs to know the column is there, not the formula behind it
- revision history and label-change notes, unless they change what can be queried
- URLs

Write plain prose. No preamble, no bullet points, no markdown. Table summaries are 1-2 \
sentences; a table that is fully described by its title plus the survey summary gets a \
short summary, not padding.

If the survey contains only ONE table there are no siblings to discriminate against, \
so the two-part split does not apply. Return survey_summary as an empty string ("") \
and put everything worth knowing into that one table's summary: what it holds, its \
collection window and reporting rules, its universe and eligibility, its grain, and \
any restriction. The CUT rules above still apply. That summary may run longer than \
two sentences.\
"""


def clean_text(value: str | None) -> str:
    """Normalize catalog text for the model: drop Excel's CR escapes, tidy blanks."""
    if not value:
        return ""
    text = _CR_ESCAPE_RE.sub("\n", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _first_env(*names: str) -> str | None:
    """First of `names` that is set and non-empty."""
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def get_azure_client() -> AzureOpenAI:
    """Build an AzureOpenAI client from the environment (fails loud if unset).

    Both spellings of the key are accepted: AZURE_OPENAI_API_KEY is what the
    OpenAI SDKs use and what this repo's .env.example shows, AZURE_OPENAI_KEY is
    what earlier versions of this script looked for.
    """
    endpoint = _first_env("AZURE_OPENAI_ENDPOINT")
    api_key = _first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    if not endpoint or not api_key:
        missing = "AZURE_OPENAI_ENDPOINT" if not endpoint else "AZURE_OPENAI_API_KEY"
        raise SystemExit(f"{missing} is not set. Add it to your .env (see .env.example).")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )


def find_tables_meta(con: psycopg.Connection) -> list[str]:
    """Return the names of the ingested tables metadata tables (e.g. 'tables23_meta')."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        "AND lower(table_name) LIKE 'tables%!_meta' ESCAPE '!' "  # '!' escapes the literal _
        "ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def meta_table_year(table: str) -> int:
    """'tables23_meta' -> 2023. Raises on anything find_tables_meta shouldn't return."""
    m = TABLES_META_RE.match(table)
    if not m:
        raise ValueError(f"not a tables metadata table: {table!r}")
    return 2000 + int(m.group(1))


def surveys_meta_name(table: str) -> str:
    """'tables23_meta' -> 'surveys23_meta', this script's own output table."""
    return f"surveys{TABLES_META_RE.match(table).group(1)}_meta"


def table_exists(con: psycopg.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            [table],
        ).fetchone()
        is not None
    )


def build_schema(tablenames: list[str]) -> dict:
    """A Structured Outputs schema naming this group's tables as literal properties.

    Strict mode forbids dynamic keys, so a free-form {tablename: summary} map is
    not expressible. The names are known before the call, though, so pinning them
    into the schema makes the model structurally unable to skip a table or invent
    one -- the grouped call's worst failure mode, removed at decode time.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["survey_summary", "tables"],
        "properties": {
            "survey_summary": {"type": "string"},
            "tables": {
                "type": "object",
                "additionalProperties": False,
                "required": list(tablenames),
                "properties": {t: {"type": "string"} for t in tablenames},
            },
        },
    }


def build_prompt(survey: str, year: int, rows: list[tuple]) -> str:
    """Render one survey group as the user message."""
    # Parity is spelled out because the catalog states even/odd collection rules
    # generically, and the model must resolve them against *this* year to say
    # whether the data is actually there.
    parity = "even" if year % 2 == 0 else "odd"
    parts = [
        f"Survey: {survey}",
        f"Year: {year} (an {parity}-numbered year)",
        f"Tables ({len(rows)}):",
        "",
    ]
    for _ctid, tablename, tabletitle, yearcoverage, description in rows:
        parts.append(f"--- {tablename}")
        parts.append(f"Title: {clean_text(tabletitle) or '(none)'}")
        parts.append(f"Year coverage: {clean_text(yearcoverage) or '(none)'}")
        parts.append("Description:")
        parts.append(clean_text(description) or "(none)")
        parts.append("")
    return "\n".join(parts)


def call_model(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    tablenames: list[str],
    use_json_schema: bool,
) -> tuple[dict, bool]:
    """One group -> parsed JSON. Returns (payload, whether json_schema was used).

    Falls back to plain JSON mode if the deployment rejects a strict schema, so an
    older model still works with the Python validation carrying the weight.
    Temperature is deliberately not sent: newer Azure deployments reject any
    non-default value.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if use_json_schema:
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "survey_group_summary",
                        "schema": build_schema(tablenames),
                        "strict": True,
                    },
                },
            )
            return json.loads(resp.choices[0].message.content), True
        except BadRequestError as e:
            print(
                f"\n  Deployment rejected strict json_schema ({e}); "
                "falling back to JSON mode for the rest of the run.",
                flush=True,
            )

    # JSON mode: valid JSON guaranteed, shape is not, so name the keys in-band.
    keys = ", ".join(repr(t) for t in tablenames)
    messages[1]["content"] += (
        f'\n\nReturn JSON: {{"survey_summary": <string>, "tables": {{...}}}} '
        f"where the keys of 'tables' are exactly these, and nothing else: {keys}."
    )
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content), False


def summarize_group(
    client: AzureOpenAI,
    deployment: str,
    survey: str,
    year: int,
    rows: list[tuple],
    use_json_schema: bool,
) -> tuple[str, dict[str, str], bool]:
    """Summarize one (year, survey) group, validating the table keys come back intact.

    A summary silently attached to the wrong sibling is the worst outcome here --
    it reads perfectly plausibly -- so a key mismatch retries once and then raises
    rather than writing anything.
    """
    prompt = build_prompt(survey, year, rows)
    tablenames = [r[1] for r in rows]
    expected = {t.lower() for t in tablenames}

    last_error = ""
    for attempt in (1, 2):
        payload, use_json_schema = call_model(
            client, deployment, prompt, tablenames, use_json_schema
        )
        tables = payload.get("tables") or {}
        got = {k.lower() for k in tables}
        if got == expected:
            by_name = {k.lower(): v for k, v in tables.items()}
            return (
                payload.get("survey_summary", "").strip(),
                {t: by_name[t.lower()].strip() for t in tablenames},
                use_json_schema,
            )
        last_error = (
            f"missing={sorted(expected - got)} unexpected={sorted(got - expected)}"
        )
        if attempt == 1:
            print(f"\n  {year} {survey}: key mismatch ({last_error}); retrying", flush=True)

    raise SystemExit(
        f"{year} {survey}: model returned the wrong table keys twice ({last_error}). "
        "Refusing to write possibly misattributed summaries."
    )


def ensure_schema(con: psycopg.Connection, tables_meta: str, surveys_meta: str) -> None:
    """Create this year's output column and survey table. Idempotent."""
    con.execute(
        sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS description_summary TEXT").format(
            sql.Identifier(tables_meta)
        )
    )
    con.execute(
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "survey VARCHAR PRIMARY KEY, surveynumber BIGINT, surveyorder BIGINT, "
            "table_count INTEGER, summary TEXT)"
        ).format(sql.Identifier(surveys_meta))
    )


def list_surveys(con: psycopg.Connection, tables_meta: str) -> list[str]:
    """Distinct surveys in one year, in the catalog's own presentation order."""
    rows = con.execute(
        sql.SQL(
            "SELECT survey FROM {} WHERE survey IS NOT NULL "
            "GROUP BY survey ORDER BY min(surveyorder), survey"
        ).format(sql.Identifier(tables_meta))
    ).fetchall()
    return [r[0] for r in rows]


def fetch_group(con: psycopg.Connection, tables_meta: str, survey: str) -> list[tuple]:
    """One survey's tables. ctid is carried because early-year meta tables have no PK."""
    return con.execute(
        sql.SQL(
            "SELECT ctid::text, tablename, tabletitle, yearcoverage, description "
            "FROM {} WHERE survey = %s ORDER BY tablenumber, tablename"
        ).format(sql.Identifier(tables_meta)),
        [survey],
    ).fetchall()


def group_is_done(
    con: psycopg.Connection, tables_meta: str, surveys_meta: str, survey: str
) -> bool:
    """True if this group was fully summarized by an earlier run."""
    if not table_exists(con, surveys_meta):
        return False
    has_survey = con.execute(
        sql.SQL("SELECT 1 FROM {} WHERE survey = %s AND summary IS NOT NULL").format(
            sql.Identifier(surveys_meta)
        ),
        [survey],
    ).fetchone()
    if not has_survey:
        return False
    pending = con.execute(
        sql.SQL(
            "SELECT 1 FROM {} WHERE survey = %s AND description_summary IS NULL LIMIT 1"
        ).format(sql.Identifier(tables_meta)),
        [survey],
    ).fetchone()
    return pending is None


def write_group(
    con: psycopg.Connection,
    tables_meta: str,
    surveys_meta: str,
    survey: str,
    rows: list[tuple],
    survey_summary: str,
    table_summaries: dict[str, str],
) -> None:
    """Persist one group's summaries. Re-runnable: the survey row upserts."""
    con.execute(
        sql.SQL(
            "INSERT INTO {} (survey, surveynumber, surveyorder, table_count, summary) "
            "SELECT %s, min(surveynumber), min(surveyorder), count(*), %s "
            "FROM {} WHERE survey = %s "
            "ON CONFLICT (survey) DO UPDATE SET "
            "surveynumber = EXCLUDED.surveynumber, surveyorder = EXCLUDED.surveyorder, "
            "table_count = EXCLUDED.table_count, summary = EXCLUDED.summary"
        ).format(sql.Identifier(surveys_meta), sql.Identifier(tables_meta)),
        [survey, survey_summary, survey],
    )
    for ctid, tablename, *_ in rows:
        con.execute(
            sql.SQL("UPDATE {} SET description_summary = %s WHERE ctid = %s::tid").format(
                sql.Identifier(tables_meta)
            ),
            [table_summaries[tablename], ctid],
        )


def csv_paths(csv_dir: Path, year: int) -> tuple[Path, Path]:
    """The (surveys, tables) CSV pair for one year, named after their DB tables."""
    yy = f"{year % 100:02d}"
    return csv_dir / f"surveys{yy}_meta.csv", csv_dir / f"tables{yy}_meta.csv"


def _merge_csv(
    path: Path, fieldnames: list[str], key_fields: tuple[str, ...], new_rows: list[dict]
) -> None:
    """Rewrite `path` with `new_rows` merged over what it already holds.

    Merged rather than appended because a run may cover a subset of surveys, or
    re-do a group whose summary was unsatisfactory; either way the file stays a
    complete, single-valued snapshot of the year. At most ~50 rows, so the
    read-modify-write costs nothing.
    """
    existing: dict[tuple, dict] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[tuple(row[k] for k in key_fields)] = row
    for row in new_rows:
        existing[tuple(row[k] for k in key_fields)] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(existing[key])


def write_group_csv(
    csv_dir: Path,
    year: int,
    survey: str,
    rows: list[tuple],
    survey_summary: str,
    table_summaries: dict[str, str],
) -> None:
    """Bank one group's model output to CSV, whether or not the DB is written."""
    surveys_csv, tables_csv = csv_paths(csv_dir, year)
    _merge_csv(
        surveys_csv,
        SURVEYS_CSV_FIELDS,
        ("survey",),
        [{"survey": survey, "table_count": str(len(rows)), "summary": survey_summary}],
    )
    _merge_csv(
        tables_csv,
        TABLES_CSV_FIELDS,
        ("survey", "tablename"),
        [
            {"survey": survey, "tablename": t, "description_summary": summary}
            for t, summary in table_summaries.items()
        ],
    )


def read_group_csv(
    csv_dir: Path, year: int, survey: str, tablenames: list[str]
) -> tuple[str, dict[str, str]]:
    """Load one group's summaries from CSV instead of calling the model.

    Fails loud on anything partial: a half-loaded group is the same
    misattribution risk as a model returning the wrong keys.
    """
    surveys_csv, tables_csv = csv_paths(csv_dir, year)
    for path in (surveys_csv, tables_csv):
        if not path.exists():
            raise SystemExit(
                f"{year} {survey}: {path} does not exist. Run without --from-csv "
                "first to generate it."
            )

    with surveys_csv.open(newline="", encoding="utf-8") as f:
        by_survey = {row["survey"]: row["summary"] for row in csv.DictReader(f)}
    if survey not in by_survey:
        raise SystemExit(f"{year} {survey}: no row for this survey in {surveys_csv}.")

    with tables_csv.open(newline="", encoding="utf-8") as f:
        by_table = {
            row["tablename"].lower(): row["description_summary"]
            for row in csv.DictReader(f)
            if row["survey"] == survey
        }
    missing = [t for t in tablenames if t.lower() not in by_table]
    if missing:
        raise SystemExit(
            f"{year} {survey}: {tables_csv} is missing summaries for {missing}. "
            "Re-run that group without --from-csv."
        )
    # An empty survey summary is legitimate: single-table groups put everything
    # in the table summary, so membership is the test, not truthiness.
    return by_survey[survey], {t: by_table[t.lower()] for t in tablenames}


def parse_years(spec: str) -> list[int]:
    """Parse a '<year>' or '<first>-<last>' selection into a list of years."""
    m = re.fullmatch(r"\s*(\d{4})\s*(?:-\s*(\d{4})\s*)?", spec)
    if not m:
        raise argparse.ArgumentTypeError(
            f"expected a four-digit year or a 'first-last' range, got {spec!r}"
        )
    first, last = int(m.group(1)), int(m.group(2) or m.group(1))
    if last < first:
        raise argparse.ArgumentTypeError(
            f"first year must not be later than last year, got {spec!r}"
        )
    return list(range(first, last + 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize IPEDS table descriptions one (year, survey) group at a time, "
            "via Azure OpenAI."
        )
    )
    p.add_argument(
        "--years",
        type=parse_years,
        help="Years to process, as '2023' or '2019-2023' (default: every year on disk).",
    )
    p.add_argument(
        "--survey",
        help="Process only this survey, e.g. 'Fall Enrollment' (matched exactly).",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Stop after this many survey groups. Useful for sampling before a full run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each prompt and the model's output; write nothing to the database.",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=(
            "Directory holding the summary CSVs written by every run and read back "
            f"by --from-csv (default: {DEFAULT_CSV_DIR})."
        ),
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the summary CSVs (model output is then only in the database).",
    )
    p.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Load summaries from --csv-dir instead of calling the model. Use this to "
            "restore summaries after a re-ingest without paying for tokens again."
        ),
    )
    p.add_argument(
        "--database-url",
        default=DATABASE_URL,
        help="Postgres connection URL (defaults to $IPEDS_DATABASE_URL, then $DATABASE_URL).",
    )
    p.add_argument(
        "--deployment",
        default=_first_env("AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_DEPLOYMENT"),
        help=(
            "Azure OpenAI deployment name (defaults to $AZURE_OPENAI_DEPLOYMENT_NAME,"
            " then $AZURE_OPENAI_DEPLOYMENT)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit(
            "IPEDS_DATABASE_URL is not set. Add it to a .env file (see .env.example) "
            "or pass --database-url."
        )
    if args.from_csv and args.no_csv:
        raise SystemExit("--from-csv and --no-csv contradict each other.")

    # --from-csv touches no model, so it needs neither a deployment nor credentials.
    client = None
    if not args.from_csv:
        if not args.deployment:
            raise SystemExit(
                "AZURE_OPENAI_DEPLOYMENT_NAME is not set. Add it to a .env file (see "
                ".env.example) or pass --deployment."
            )
        client = get_azure_client()
    con = psycopg.connect(args.database_url)
    con.autocommit = True  # commit each group so an interrupted run keeps its progress

    metas = find_tables_meta(con)
    if not metas:
        raise SystemExit(
            "No 'tables*_meta' tables found. Run ingest.py first (and check the URL)."
        )
    if args.years:
        wanted = set(args.years)
        metas = [m for m in metas if meta_table_year(m) in wanted]
        if not metas:
            raise SystemExit(f"No tables*_meta tables for year(s) {args.years}.")

    # Enumerated up front so --limit and the progress bar both know the real total.
    groups: list[tuple[str, int, str]] = []
    for meta in metas:
        for survey in list_surveys(con, meta):
            if args.survey and survey != args.survey:
                continue
            groups.append((meta, meta_table_year(meta), survey))
    if not groups:
        raise SystemExit(
            f"No survey groups matched (survey={args.survey!r}, years={args.years})."
        )

    if not args.dry_run:
        for meta in {m for m, _, _ in groups}:
            ensure_schema(con, meta, surveys_meta_name(meta))

    if args.limit:
        groups = groups[: args.limit]

    use_json_schema = True
    processed = skipped = 0
    for meta, year, survey in tqdm(groups, desc="survey groups", unit="grp"):
        surveys_meta = surveys_meta_name(meta)
        if not args.dry_run and group_is_done(con, meta, surveys_meta, survey):
            skipped += 1
            continue

        rows = fetch_group(con, meta, survey)
        if not rows:
            continue

        if args.dry_run:
            print(f"\n{'=' * 78}\n{year} / {survey} ({len(rows)} tables)\n{'=' * 78}")
            if not args.from_csv:
                print(f"\n--- PROMPT ---\n{build_prompt(survey, year, rows)}")

        if args.from_csv:
            survey_summary, table_summaries = read_group_csv(
                args.csv_dir, year, survey, [r[1] for r in rows]
            )
        else:
            survey_summary, table_summaries, use_json_schema = summarize_group(
                client, args.deployment, survey, year, rows, use_json_schema
            )
            if not args.no_csv:
                write_group_csv(
                    args.csv_dir, year, survey, rows, survey_summary, table_summaries
                )

        if args.dry_run:
            print(f"--- SURVEY SUMMARY ---\n{survey_summary}\n")
            print("--- TABLE SUMMARIES ---")
            for tablename, summary in table_summaries.items():
                print(f"\n[{tablename}]\n{summary}")
            print()
        else:
            write_group(
                con, meta, surveys_meta, survey, rows, survey_summary, table_summaries
            )
        processed += 1

    con.close()
    if args.from_csv:
        mode = f"summaries replayed from {args.csv_dir}"
    else:
        mode = "structured outputs" if use_json_schema else "JSON mode (schema unenforced)"
        if not args.no_csv:
            mode += f", banked to {args.csv_dir}"
    verb = "Would have written" if args.dry_run else "Summarized"
    print(
        f"\nDone. {verb} {processed} survey group(s), skipped {skipped} already done, "
        f"using {mode}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
