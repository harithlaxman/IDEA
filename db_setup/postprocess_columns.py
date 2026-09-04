"""Post-process the ingested Postgres DB: LLM-summarize *column* descriptions.

The sibling `postprocess.py` handles table descriptions. This handles the much
larger problem underneath them: `vartable<YY>_meta.longdescription` is 33,871 rows
and 26.5M characters across the 20 years, and one call to the agent's
`get_column_descriptions_of_table(2023, 'sfa2223_p1')` hands back 555,065 of those
characters -- about 139,000 tokens for a single tool result.

Two facts about the corpus shape the design:

  * It repeats itself across years. Deduping on the description text with year
    tokens masked ('2022-23' -> 'YYYY') takes 33,871 rows down to 3,714 distinct
    texts, because a variable's wording is near-static and revised occasionally
    rather than rewritten annually. The unit of work here is therefore a distinct
    text, not a row: summarize once, write the result to every row that shares it.
  * The weight is concentrated. Descriptions of 800+ characters are 32% of rows
    but 73% of the characters, so the threshold buys most of the payload back for
    a third of the calls. Shorter descriptions are left alone and the reader
    coalesces to the raw text.

Because one summary is reused across every year that shares the text, the
summaries are written year-generically -- the concrete year reaches the reader
through the variable title and the table name instead.

Every summary is banked to CSV (see --csv-dir) so a rebuilt database can be
restored with --from-csv without paying for the tokens again.

Config (from environment / a local .env):
    IPEDS_DATABASE_URL            Postgres URL; DATABASE_URL is accepted as a fallback.
    AZURE_OPENAI_ENDPOINT         e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY          Azure OpenAI API key; AZURE_OPENAI_KEY also accepted.
    AZURE_OPENAI_DEPLOYMENT_NAME  Deployment (model) name to call.
    AZURE_OPENAI_API_VERSION      Optional; defaults to a recent stable version.
"""

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from openai import AzureOpenAI, BadRequestError
from psycopg import sql
from tqdm import tqdm

from postprocess import (
    DATABASE_URL,
    DEFAULT_CSV_DIR,
    _first_env,
    _merge_csv,
    clean_text,
    get_azure_client,
    parse_years,
)

# Matches the variable catalogs this script reads, e.g. 'vartable23_meta'.
VARTABLE_META_RE = re.compile(r"^vartable(\d{2})_meta$", re.IGNORECASE)

# Dates are masked out of the dedupe key so that the same wording in different
# years collapses to one unit of work. '2023', '2022-23' and '1995' all match.
_YEAR_RE = re.compile(r"(19|20)\d{2}(-\d{2})?")
_WS_RE = re.compile(r"\s+")

# Bumped whenever norm_key changes. Banked CSV rows carry it, so a normalization
# change shows up as a version mismatch rather than as a silent flood of keys
# that suddenly look unbanked.
NORM_VERSION = "1"

# Below this masked length a description is already about a paragraph and
# compresses poorly, so it is left raw rather than spending a call on it.
DEFAULT_MIN_CHARS = 800

# Summaries are flushed to CSV in batches during the run rather than at the end:
# a crash 1,400 calls into a 1,412-call run must not throw away the tokens already
# spent. _merge_csv rewrites the whole file, so batching keeps that cheap.
CSV_FLUSH_EVERY = 50

COLUMNS_CSV_NAME = "columns.csv"
COLUMNS_CSV_FIELDS = [
    "norm_version",
    "key_sha256",
    "char_len",
    "row_count",
    "sample_tablename",
    "sample_varname",
    "summary",
]

SYSTEM_PROMPT = """\
You write navigation metadata for an AI agent that explores the IPEDS database by \
choosing which columns to query.

You are given one IPEDS variable: its name, its title, an example table it appears in, \
and its full description from the codebook. That description was written for a human \
reading a PDF, so it spends most of its length on derivation arithmetic and reporting \
mechanics. Condense it to what changes how the agent uses the column.

The dates in the description have been replaced with YYYY, because the identical text \
is reused across many collection years and your summary will be stored against all of \
them. Write the summary so it is true in every year: never emit YYYY, and never name a \
specific year or academic year. The concrete year already reaches the reader through the \
variable's title and the table's name.

KEEP:
- what the variable actually measures, in plain terms
- its unit and denominator -- dollars, headcount, FTE, a percentage and of what base
- the population it covers: full-time vs part-time, first-time, degree/certificate-seeking, \
undergraduate vs graduate, in-state vs out-of-state
- any condition that decides whether the column is populated at all for an institution: \
institutional control (public / private not-for-profit / for-profit), calendar system \
(standard term vs continuous vs program-length), whether the institution reports on a fall \
cohort or a full 12-month aid year, degree-granting status. This is the single most useful \
thing in most of these descriptions -- a column that is blank for half the universe is a \
trap, and the agent must be able to see the restriction before it writes the query.
- how it differs from an obviously adjacent variable, when the description says so

CUT:
- step-by-step derivation arithmetic. "Generated by subtracting the average grant aid from \
the total cost of attendance, where total cost of attendance is the sum of published tuition \
and required fees, books and supplies, and the weighted average of room and board and other \
expenses" is one clause: an average net price. The agent needs to know the column exists and \
what it means, not how NCES computed it.
- imputation notes, response rates, and data-quality commentary
- revision history and label-change notes, unless they change what can be queried
- URLs
- restating the variable's title. It is stored separately and served alongside your summary.

Write prose, no preamble and no bullets, 1 to 3 sentences. A variable that its title \
already explains should get one short sentence, not padding. Return JSON: \
{"summary": <string>}.
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary"],
    "properties": {"summary": {"type": "string"}},
}


def norm_key(value: str | None) -> str:
    """Dedupe key: cleaned text, whitespace collapsed, year tokens masked."""
    return _YEAR_RE.sub("YYYY", _WS_RE.sub(" ", clean_text(value)).strip())


def key_hash(key: str) -> str:
    """Stable join key for the CSV bank; the masked text itself is too big to store."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class Group:
    """One distinct masked description and every row that carries it."""

    key: str
    sample_tablename: str = ""
    sample_varname: str = ""
    sample_vartitle: str = ""
    # meta table -> ctids of the rows still needing a summary
    pending: dict[str, list[str]] = field(default_factory=dict)
    row_count: int = 0
    done_count: int = 0

    @property
    def is_done(self) -> bool:
        return not self.pending


def find_vartable_meta(con: psycopg.Connection) -> list[str]:
    """Return the names of the ingested variable metadata tables."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        "AND lower(table_name) LIKE 'vartable%!_meta' ESCAPE '!' "  # '!' escapes the literal _
        "ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def meta_table_year(table: str) -> int:
    """'vartable23_meta' -> 2023. Raises on anything find_vartable_meta shouldn't return."""
    m = VARTABLE_META_RE.match(table)
    if not m:
        raise ValueError(f"not a variable metadata table: {table!r}")
    return 2000 + int(m.group(1))


def column_exists(con: psycopg.Connection, table: str, column: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
            [table, column],
        ).fetchone()
        is not None
    )


def ensure_schema(con: psycopg.Connection, vartable_meta: str) -> None:
    """Create this year's output column. Idempotent."""
    con.execute(
        sql.SQL(
            "ALTER TABLE {} ADD COLUMN IF NOT EXISTS longdescription_summary TEXT"
        ).format(sql.Identifier(vartable_meta))
    )


def collect_groups(
    con: psycopg.Connection, metas: list[str], min_chars: int
) -> list[Group]:
    """Scan every year's variable catalog and group the long descriptions by masked text.

    ctid is carried because the vartable catalogs have no primary key, the same
    reason postprocess.py updates tables<YY>_meta by ctid.
    """
    groups: dict[str, Group] = {}
    for meta in metas:
        # The output column is absent under --dry-run, which never creates it.
        has_summary = column_exists(con, meta, "longdescription_summary")
        summary_expr = "longdescription_summary" if has_summary else "NULL::text"
        rows = con.execute(
            sql.SQL(
                "SELECT ctid::text, tablename, varname, vartitle, longdescription, "
                + summary_expr
                + " FROM {}"
            ).format(sql.Identifier(meta))
        ).fetchall()
        for ctid, tablename, varname, vartitle, longdescription, summary in rows:
            key = norm_key(longdescription)
            if len(key) < min_chars:
                continue
            g = groups.get(key)
            if g is None:
                g = groups[key] = Group(
                    key=key,
                    sample_tablename=tablename or "",
                    sample_varname=varname or "",
                    sample_vartitle=clean_text(vartitle),
                )
            g.row_count += 1
            if summary is None:
                g.pending.setdefault(meta, []).append(ctid)
            else:
                g.done_count += 1

    # Highest-reuse texts first, so --limit samples the ones that matter and an
    # interrupted run has already bought the biggest payload reduction.
    return sorted(
        groups.values(), key=lambda g: (-g.row_count, -len(g.key), g.sample_varname)
    )


def build_prompt(group: Group) -> str:
    """Render one distinct description as the user message."""
    return "\n".join(
        [
            f"Variable: {group.sample_varname}",
            f"Title: {group.sample_vartitle or '(none)'}",
            f"Example table: {group.sample_tablename or '(none)'}",
            "Description:",
            group.key,
        ]
    )


def call_model(
    client: AzureOpenAI, deployment: str, prompt: str, use_json_schema: bool
) -> tuple[str, bool]:
    """One description -> its summary. Returns (summary, whether json_schema was used).

    Falls back to plain JSON mode if the deployment rejects a strict schema, so an
    older model still works. Temperature is deliberately not sent: newer Azure
    deployments reject any non-default value.
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
                        "name": "column_summary",
                        "schema": SUMMARY_SCHEMA,
                        "strict": True,
                    },
                },
            )
            return _read_summary(resp), True
        except BadRequestError as e:
            print(
                f"\n  Deployment rejected strict json_schema ({e}); "
                "falling back to JSON mode for the rest of the run.",
                flush=True,
            )

    resp = client.chat.completions.create(
        model=deployment, messages=messages, response_format={"type": "json_object"}
    )
    return _read_summary(resp), False


def _read_summary(resp) -> str:
    """Pull the summary out of a response, failing loud on an empty one.

    An empty summary would be written over the raw description's place in the
    coalesce and leave the agent with nothing, so it is an error rather than a
    value.
    """
    summary = (json.loads(resp.choices[0].message.content).get("summary") or "").strip()
    if not summary:
        raise SystemExit("Model returned an empty summary; refusing to write it.")
    return summary


def columns_csv_path(csv_dir: Path) -> Path:
    return csv_dir / COLUMNS_CSV_NAME


def write_csv_rows(csv_dir: Path, rows: list[dict]) -> None:
    """Bank summaries to CSV, merged over what the file already holds."""
    _merge_csv(
        columns_csv_path(csv_dir),
        COLUMNS_CSV_FIELDS,
        ("norm_version", "key_sha256"),
        rows,
    )


def csv_row(group: Group, summary: str) -> dict:
    return {
        "norm_version": NORM_VERSION,
        "key_sha256": key_hash(group.key),
        "char_len": str(len(group.key)),
        "row_count": str(group.row_count),
        "sample_tablename": group.sample_tablename,
        "sample_varname": group.sample_varname,
        "summary": summary,
    }


def read_csv_summaries(csv_dir: Path, groups: list[Group]) -> dict[str, str]:
    """Load summaries for `groups` from CSV instead of calling the model.

    Validated all at once, before anything is written: a half-loaded replay would
    leave the catalog in a state no later run can tell apart from a genuinely
    interrupted one.
    """
    path = columns_csv_path(csv_dir)
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run without --from-csv first to generate it."
        )

    banked: dict[str, str] = {}
    other_versions: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["norm_version"] == NORM_VERSION:
                banked[row["key_sha256"]] = row["summary"]
            else:
                other_versions.add(row["norm_version"])

    found = {g.key: banked[key_hash(g.key)] for g in groups if key_hash(g.key) in banked}
    missing = [g for g in groups if g.key not in found]
    if missing:
        examples = ", ".join(
            f"{g.sample_tablename}.{g.sample_varname}" for g in missing[:5]
        )
        hint = (
            f" The file also holds rows normalized as version(s) "
            f"{sorted(other_versions)}, which no longer match norm_key; those need "
            "regenerating."
            if other_versions
            else ""
        )
        raise SystemExit(
            f"{path} is missing summaries for {len(missing)} of {len(groups)} "
            f"descriptions (e.g. {examples}).{hint} Re-run without --from-csv."
        )
    return found


def write_group(con: psycopg.Connection, group: Group, summary: str) -> int:
    """Write one summary to every pending row that shares the description."""
    written = 0
    for meta, ctids in group.pending.items():
        cur = con.execute(
            sql.SQL(
                "UPDATE {} SET longdescription_summary = %s "
                "WHERE ctid = ANY(%s::text[]::tid[])"
            ).format(sql.Identifier(meta)),
            [summary, ctids],
        )
        written += cur.rowcount
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize the large IPEDS column descriptions, one call per distinct "
            "description rather than per row, via Azure OpenAI."
        )
    )
    p.add_argument(
        "--years",
        type=parse_years,
        help="Years to process, as '2023' or '2004-2023' (default: every year on disk).",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=(
            "Only summarize descriptions at least this long once normalized "
            f"(default: {DEFAULT_MIN_CHARS})."
        ),
    )
    p.add_argument(
        "--table",
        help="Only descriptions appearing in this table, e.g. 'sfa2223_p1' (matched exactly).",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Stop after this many distinct descriptions. Useful for sampling first.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each prompt and the model's output; write nothing to the database.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel model calls (default: 4). Database writes stay single-threaded.",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=(
            "Directory holding the summary CSV written by every run and read back "
            f"by --from-csv (default: {DEFAULT_CSV_DIR})."
        ),
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the summary CSV (model output is then only in the database).",
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

    metas = find_vartable_meta(con)
    if not metas:
        raise SystemExit(
            "No 'vartable*_meta' tables found. Run ingest.py first (and check the URL)."
        )
    if args.years:
        wanted = set(args.years)
        metas = [m for m in metas if meta_table_year(m) in wanted]
        if not metas:
            raise SystemExit(f"No vartable*_meta tables for year(s) {args.years}.")

    if not args.dry_run:
        for meta in metas:
            ensure_schema(con, meta)

    groups = collect_groups(con, metas, args.min_chars)
    if args.table:
        wanted_table = args.table.strip().lower()
        # Re-scan rather than filter on sample_tablename: a description shared by
        # several tables keeps whichever sample it happened to be created with.
        keys = {
            norm_key(r[0])
            for meta in metas
            for r in con.execute(
                sql.SQL("SELECT longdescription FROM {} WHERE lower(tablename) = %s")
                .format(sql.Identifier(meta)),
                [wanted_table],
            ).fetchall()
        }
        groups = [g for g in groups if g.key in keys]

    pending = groups if args.dry_run else [g for g in groups if not g.is_done]
    skipped = len(groups) - len(pending)
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"Nothing to do: {skipped} description(s) already summarized.")
        return

    def iter_summaries():
        """Yield (group, summary) as each becomes available, in work-list order."""
        if args.from_csv:
            banked = read_csv_summaries(args.csv_dir, pending)
            yield from ((g, banked[g.key]) for g in pending)
            return
        # The first call is made alone so a deployment that rejects strict
        # json_schema is detected once, and the pool then runs on the settled
        # mode instead of every worker discovering it separately.
        summary, use_json_schema = call_model(
            client, args.deployment, build_prompt(pending[0]), True
        )
        yield pending[0], summary
        rest = pending[1:]
        if not rest:
            return
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            results = pool.map(
                lambda g: call_model(
                    client, args.deployment, build_prompt(g), use_json_schema
                ),
                rest,
            )
            for group, (summary, _) in zip(rest, results):
                yield group, summary

    bank = not args.no_csv and not args.from_csv
    csv_batch: list[dict] = []
    rows_written = 0
    for group, summary in tqdm(
        iter_summaries(), total=len(pending), desc="descriptions", unit="desc"
    ):
        # Banked before the write, and on dry runs too: the tokens are already
        # spent, and a sampling run whose output is approved should not have to
        # be paid for a second time.
        if bank:
            csv_batch.append(csv_row(group, summary))
            if len(csv_batch) >= CSV_FLUSH_EVERY:
                write_csv_rows(args.csv_dir, csv_batch)
                csv_batch.clear()
        if args.dry_run:
            print(f"\n{'=' * 78}")
            print(
                f"{group.sample_tablename}.{group.sample_varname} "
                f"({len(group.key)} chars, {group.row_count} rows)"
            )
            print("=" * 78)
            if not args.from_csv:
                print(f"\n--- PROMPT ---\n{build_prompt(group)}")
            print(f"\n--- SUMMARY ({len(summary)} chars) ---\n{summary}")
            continue
        rows_written += write_group(con, group, summary)
    if csv_batch:
        write_csv_rows(args.csv_dir, csv_batch)

    con.close()
    source = (
        f"replayed from {columns_csv_path(args.csv_dir)}"
        if args.from_csv
        else "dry run, nothing written"
        if args.dry_run
        else f"banked to {columns_csv_path(args.csv_dir)}"
    )
    print(
        f"\nDone: {len(pending)} description(s) summarized, {skipped} already done, "
        f"{rows_written} catalog row(s) updated ({source})."
    )


if __name__ == "__main__":
    main()
