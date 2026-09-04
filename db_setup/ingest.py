"""Download IPEDS releases and ingest them into Postgres.

Each academic year ships as one zip holding an Access DB plus a data dictionary
workbook, and both are kept: `--mode download` fetches into data/zips/ and
extracts to data/accdb/, `--mode ingest` loads what is there, `--mode all` does
both (default).

Data tables come from the .accdb, typed from `mdb-schema` rather than CSV
sampling, so a column keeps one type across years. Metadata catalogs come from
the workbook, *not* the .accdb: NCES revises only the workbook, so the accdb
catalogs carry stale year labels and lack columns (varNumber, varOrder,
Tablenumber).

Table and column names are lowercased, metadata tables gain a '_meta' suffix,
and DRV*/DFR* rollups are skipped along with their catalog rows.
"""

import argparse
import io
import os
import re
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import date

import openpyxl
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from tqdm import tqdm

# Load configuration (DATABASE_URL, ...) from a local .env if one is present.
load_dotenv()

IPEDS_DATABASE_URL = os.environ.get("IPEDS_DATABASE_URL")

# Defaults anchored to this script's location: <project_root>/data/{accdb,zips}.
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
ACCDB_PATH = os.path.join(_DATA_DIR, "accdb")
ZIPS_PATH = os.path.join(_DATA_DIR, "zips")

# NCES publishes one zip per academic year, named for the span it covers:
# .../zipfiles/IPEDS_2019-20_Final.zip. 2004-05 is the earliest available.
IPEDS_BASE_URL = "https://nces.ed.gov/ipeds/tablefiles/zipfiles"
FIRST_IPEDS_YEAR = 2004
# nces.ed.gov serves the zips only to a browser-like client.
USER_AGENT = "Mozilla/5.0 (compatible; ipeds-ingest/1.0)"
# nces.ed.gov stalls mid-transfer often, so a read quiet this long is retried.
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_ATTEMPTS = 3

# Fixed date format we ask mdb-export to emit; matches Postgres timestamp input.
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Access type (as printed by `mdb-schema`, default backend) -> Postgres type.
TYPE_MAP = {
    "Text": sql.SQL("VARCHAR"),
    "Memo/Hyperlink": sql.SQL("VARCHAR"),
    "GUID": sql.SQL("VARCHAR"),
    "Replication ID": sql.SQL("VARCHAR"),
    "Boolean": sql.SQL("BOOLEAN"),
    "Byte": sql.SQL("SMALLINT"),
    "Integer": sql.SQL("INTEGER"),  # Access "Integer" is 16-bit; widen to be safe
    "Long Integer": sql.SQL("INTEGER"),
    "Single": sql.SQL("REAL"),
    "Double": sql.SQL("DOUBLE PRECISION"),
    "Float": sql.SQL("DOUBLE PRECISION"),
    "Currency": sql.SQL("NUMERIC(19,4)"),
    "Numeric": sql.SQL("DOUBLE PRECISION"),
    "Decimal": sql.SQL("DOUBLE PRECISION"),
    "DateTime": sql.SQL("TIMESTAMP"),
    "DateTime (Short)": sql.SQL("TIMESTAMP"),
    "OLE": sql.SQL("BYTEA"),
    "Binary": sql.SQL("BYTEA"),
}

# Catalog tables carried inside the .accdb. Superseded by the workbook sheets
# below, so they are skipped rather than ingested.
METADATA_TABLES = [
    "tables",
    "valuesets",
    "sectiontable",
    "filenames",
    "vartable",
    "newvariables",
]

# Workbook sheets ingested as the metadata catalogs, named '<kind><YY>' (e.g.
# 'Tables19'). Casing drifts across years, so they are matched
# case-insensitively. 'newvariables' exists only some years; the rest are required.
META_SHEETS = ("tables", "vartable", "valuesets", "newvariables")
OPTIONAL_SHEETS = {"newvariables"}

# The data dictionary workbook, e.g. IPEDS201920TablesDoc.xlsx. Group 1 is the
# survey start year; its last two digits are the YY sheet names carry.
TABLESDOC_RE = re.compile(r"^IPEDS(\d{4})(\d{2})Tablesdoc\.xlsx$", re.IGNORECASE)

# Left VARCHAR even when a year's values are all numeric: callers regex-match
# codevalue to sort it, and `~` has no operator for a numeric column.
FORCE_TEXT_COLUMNS = {"codevalue"}

# NCES-computed rollups (DRV*, older DFR*). Not source data; skipped on ingest.
EXCLUDED_PREFIXES = ("DRV", "DFR")

# Metadata table: METADATA_TABLES + two-digit year (e.g. 'Tables23'), optionally
# with the '_RV' revised-data suffix. Anchored so data tables aren't caught.
_META_KINDS = "|".join(map(re.escape, METADATA_TABLES))
META_RE = re.compile(rf"^(?:{_META_KINDS})\d{{2}}(?:_RV)?$", re.IGNORECASE)

# Matches lines like:   [COLUMN NAME]		Text (510),
COL_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s+(?P<type>.+?)\s*,?\s*$")


def access_to_pg_type(access_type: str) -> sql.SQL:
    """Map an Access type token, ignoring size '(...)' and a 'NOT NULL' suffix."""
    base = re.sub(r"\s*\(.*?\)", "", access_type)  # drop size, e.g. Text (510)
    base = re.sub(r"\s+NOT\s+NULL\s*$", "", base, flags=re.IGNORECASE)  # drop NOT NULL
    base = base.strip()
    if base not in TYPE_MAP:
        raise ValueError(f"Unmapped Access type: {access_type!r}")
    return TYPE_MAP[base]


def get_schema(db_path: str, table: str) -> dict[str, sql.SQL]:
    """Return {column_name: postgres_type} for one table, in column order."""
    out = subprocess.run(
        ["mdb-schema", "-T", table, db_path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    # Keep only the lines between the opening '(' and the closing ');'.
    cols: dict[str, sql.SQL] = {}
    in_body = False
    for line in out.splitlines():
        if not in_body:
            if line.strip().startswith("("):
                in_body = True
            continue
        if line.strip().startswith(")"):
            break
        m = COL_RE.match(line)
        if m:
            cols[m.group("name")] = access_to_pg_type(m.group("type"))
    if not cols:
        raise ValueError(f"No columns parsed for {table} in {db_path}")
    return cols


def column_defs(cols: dict[str, sql.SQL]) -> sql.Composed:
    """Render CREATE TABLE column definitions. Names are lowercased (so clients
    write unquoted SQL) and quoted via sql.Identifier."""
    return sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name.lower()), dtype)
        for name, dtype in cols.items()
    )


def banner(text: str) -> None:
    """Phase header, so download and ingest output are easy to tell apart."""
    print(f"\n=== {text} ===", flush=True)


def release_name(year: int) -> str:
    """NCES names a release for the span it covers, e.g. 2019 -> '2019-20'."""
    return f"{year}-{(year + 1) % 100:02d}"


def _unlink(path: str) -> None:
    """Remove a partial download, ignoring one that was never created."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _copy_with_progress(src, dst, total: int, desc: str, initial: int = 0) -> None:
    """Copy a stream in 1 MB chunks behind a byte-scaled progress bar."""
    with tqdm(
        total=total or None,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        leave=False,
    ) as bar:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
            bar.update(len(chunk))


def _fetch_to_file(url: str, part_path: str, desc: str) -> None:
    """Download `url` into `part_path`, retrying through stalled transfers.

    Bytes on disk resume via a Range request when the server honours one (HTTP
    206), else the file restarts. HTTP errors are raised to the caller, which
    tells a missing year from a failure.
    """
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                resuming = response.status == 206
                if not resuming:
                    have = 0  # server ignored the Range; start the file over
                # On a resume Content-Length counts only the remaining bytes.
                total = int(response.headers.get("Content-Length") or 0) + have
                with open(part_path, "ab" if resuming else "wb") as out:
                    _copy_with_progress(response, out, total, desc, initial=have)
            return
        except urllib.error.HTTPError as e:
            # A complete .part makes the server reject the Range; drop it and retry.
            if e.code == 416 and have:
                _unlink(part_path)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            if attempt < DOWNLOAD_ATTEMPTS:
                tqdm.write(
                    f"{desc.strip()}: {e}; retrying ({attempt}/{DOWNLOAD_ATTEMPTS - 1})"
                )
    raise last_error if last_error else RuntimeError(f"could not download {url}")


def extract_accdb(
    zip_path: str, accdb_dir: str, label: str, force: bool = False
) -> tuple[str, bool]:
    """Extract the .accdb member of a release zip.

    Returns (file name, whether it was written). One already on disk is left
    alone unless `force`, so a re-run doesn't rewrite gigabytes.
    """
    with zipfile.ZipFile(zip_path) as z:
        # Some zips nest their files in a folder, so flatten to the basename.
        members = [m for m in z.infolist() if m.filename.lower().endswith(".accdb")]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one .accdb in {os.path.basename(zip_path)}, "
                f"found {[m.filename for m in members]}"
            )
        member = members[0]
        name = os.path.basename(member.filename)
        target = os.path.join(accdb_dir, name)
        if os.path.exists(target) and not force:
            return name, False
        with z.open(member) as src, open(target, "wb") as dst:
            _copy_with_progress(src, dst, member.file_size, f"  {label} accdb")
    return name, True


def download_release(year: int, zips_dir: str, accdb_dir: str, force: bool) -> str:
    """Fetch one year's release zip and extract its .accdb.

    Both are kept: the zip carries the dictionary workbook, the .accdb the data
    tables. Returns a status word for the run summary; a year that isn't
    published yet is not a failure.
    """
    label = release_name(year)
    zip_name = f"IPEDS_{label}_Final.zip"
    zip_path = os.path.join(zips_dir, zip_name)

    fetched = False
    if force or not os.path.exists(zip_path):
        # Renamed only once complete, so an interrupted run never leaves a
        # truncated zip that later looks finished.
        part_path = zip_path + ".part"
        try:
            _fetch_to_file(f"{IPEDS_BASE_URL}/{zip_name}", part_path, f"  {label} zip")
        except urllib.error.HTTPError as e:
            _unlink(part_path)
            if e.code == 404:
                tqdm.write(f"  {label}: not published yet (HTTP 404), skipping")
                return "missing"
            tqdm.write(f"  {label}: download failed (HTTP {e.code} {e.reason})")
            return "failed"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # The .part is left behind on purpose: a re-run resumes from it.
            tqdm.write(f"  {label}: download failed ({e}); re-run to resume")
            return "failed"
        os.replace(part_path, zip_path)
        fetched = True

    # A zip alone is no use, so a year missing its .accdb still extracts.
    try:
        accdb, written = extract_accdb(zip_path, accdb_dir, label, force=fetched)
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        tqdm.write(f"  {label}: {e}")
        return "failed"

    if fetched:
        tqdm.write(f"  {label}: saved {zip_name}, extracted {accdb}")
        return "downloaded"
    if written:
        tqdm.write(f"  {label}: zip already present, extracted {accdb}")
        return "extracted"
    tqdm.write(f"  {label}: zip and {accdb} already present, skipping")
    return "skipped"


def download_releases(
    years: list[int], zips_dir: str, accdb_dir: str, force: bool
) -> None:
    """Fetch every requested year, carrying on past the ones that fail."""
    banner(f"Downloading IPEDS releases {years[0]}-{years[-1]}")
    print(f"    zips  -> {zips_dir}", flush=True)
    print(f"    accdb -> {accdb_dir}", flush=True)
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(accdb_dir, exist_ok=True)

    tally: dict[str, int] = {}
    for year in tqdm(years, desc="downloading", unit="yr"):
        status = download_release(year, zips_dir, accdb_dir, force)
        tally[status] = tally.get(status, 0) + 1

    print(
        f"Downloaded {tally.get('downloaded', 0)} of {len(years)} year(s): "
        f"{tally.get('extracted', 0)} extracted from a kept zip, "
        f"{tally.get('skipped', 0)} already present, "
        f"{tally.get('missing', 0)} not published, "
        f"{tally.get('failed', 0)} failed.",
        flush=True,
    )
    if tally.get("failed"):
        raise SystemExit("Some downloads failed; re-run to retry them.")


def find_tablesdoc(zips_dir: str) -> dict[str, tuple[str, str]]:
    """Map two-digit year -> (zip path, entry name) for each dictionary workbook."""
    docs: dict[str, tuple[str, str]] = {}
    for name in sorted(os.listdir(zips_dir)):
        if not name.lower().endswith(".zip"):
            continue
        path = os.path.join(zips_dir, name)
        with zipfile.ZipFile(path) as z:
            # Some zips nest everything under a folder, so match on the basename.
            hits = [
                n for n in z.namelist() if TABLESDOC_RE.match(os.path.basename(n))
            ]
        if len(hits) != 1:
            raise SystemExit(
                f"Expected exactly one IPEDS*Tablesdoc.xlsx in {path}, found {hits}."
            )
        yy = TABLESDOC_RE.match(os.path.basename(hits[0])).group(1)[2:]
        if yy in docs:
            raise SystemExit(
                f"Two zips carry a 20{yy} dictionary: {docs[yy][0]} and {path}."
            )
        docs[yy] = (path, hits[0])
    if not docs:
        raise SystemExit(f"No IPEDS zips found in {zips_dir}.")
    return docs


def read_sheet(ws) -> tuple[list[str], list[list]]:
    """Return (lowercased column names, rows) for one worksheet.

    Filtered sheets extend past their data, so headerless trailing columns and
    fully blank rows are dropped.
    """
    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter, None)
    if header is None:
        raise SystemExit(f"Sheet {ws.title!r} is empty.")
    keep = [i for i, h in enumerate(header) if h is not None and str(h).strip()]
    cols = [str(header[i]).strip().lower() for i in keep]
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        raise SystemExit(f"Sheet {ws.title!r} has duplicate columns: {dupes}.")

    rows: list[list] = []
    for row in rows_iter:
        values = [row[i] if i < len(row) else None for i in keep]
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
            continue
        rows.append(values)
    return cols, rows


def infer_pg_type(values: list) -> str:
    """Pick a Postgres type from the Python types openpyxl produced.

    Excel carries no schema, so the values decide. Anything mixed or
    non-numeric — dates and booleans included — falls back to VARCHAR.
    """
    # bool has its own type() and so never satisfies the int-only check below.
    kinds = {type(v) for v in values if v is not None and v != ""}
    if kinds and kinds <= {int}:
        return "BIGINT"  # BIGINT over INTEGER: no width to reason about
    if kinds and kinds <= {int, float}:
        return "DOUBLE PRECISION"
    return "VARCHAR"


def coerce(value, is_text: bool):
    """Normalize one cell for COPY: blanks become NULL, text columns become str."""
    if value is None:
        return None
    if is_text:
        text = value.strip() if isinstance(value, str) else str(value)
        return text or None
    return None if value == "" else value


def normalize_catalog(con: psycopg.Connection, dest: str, cols: set[str]) -> None:
    """Align a metadata catalog with the ingested data tables.

    Drops rows advertising the skipped DRV*/DFR* tables, and lowercases
    tablename/varname so the catalog's values are usable as SQL directly.
    """
    if "tablename" in cols:
        # upper() guards against casing drift in the catalog itself.
        con.execute(
            sql.SQL("DELETE FROM {} WHERE upper(tablename) LIKE ANY(%s)").format(
                sql.Identifier(dest)
            ),
            [[p + "%" for p in EXCLUDED_PREFIXES]],
        )
    for column in ("tablename", "varname"):
        if column in cols:
            con.execute(
                sql.SQL(
                    "UPDATE {t} SET {c} = lower({c}) "
                    "WHERE {c} IS NOT NULL AND {c} != lower({c})"
                ).format(t=sql.Identifier(dest), c=sql.Identifier(column))
            )


def ingest_metadata(con: psycopg.Connection, zips_dir: str, drop_existing: bool,
                    seen: dict[str, str]) -> None:
    """Load each year's catalog sheets out of its dictionary workbook."""
    banner(f"Ingesting metadata catalogs from {zips_dir}")
    docs = find_tablesdoc(zips_dir)
    print(f"    {len(docs)} dictionary workbook(s) found", flush=True)
    for yy, (zip_path, entry) in tqdm(sorted(docs.items()), desc="dictionaries"):
        with zipfile.ZipFile(zip_path) as z:
            # openpyxl needs a seekable stream, so read the ~1 MB workbook
            # into memory rather than streaming it out of the zip.
            book = io.BytesIO(z.read(entry))
        wb = openpyxl.load_workbook(book, read_only=True, data_only=True)
        try:
            by_lower = {name.lower(): name for name in wb.sheetnames}
            for kind in META_SHEETS:
                sheet = by_lower.get(f"{kind}{yy}")
                if sheet is None:
                    if kind in OPTIONAL_SHEETS:
                        continue
                    raise SystemExit(
                        f"{os.path.basename(entry)} has no {kind}{yy} sheet "
                        f"(sheets: {wb.sheetnames})."
                    )
                dest = f"{kind}{yy}_meta"
                if dest in seen:
                    raise SystemExit(
                        f"Duplicate table name {dest!r} from {zip_path} "
                        f"(first seen in {seen[dest]}). Resolve naming before ingest."
                    )
                seen[dest] = zip_path

                cols, rows = read_sheet(wb[sheet])
                types = [
                    "VARCHAR"
                    if name in FORCE_TEXT_COLUMNS
                    else infer_pg_type([r[j] for r in rows])
                    for j, name in enumerate(cols)
                ]
                is_text = [t == "VARCHAR" for t in types]

                if drop_existing:
                    con.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(dest))
                    )
                con.execute(
                    sql.SQL("CREATE TABLE {} ({})").format(
                        sql.Identifier(dest),
                        column_defs({c: sql.SQL(t) for c, t in zip(cols, types)}),
                    )
                )
                with con.cursor() as cur, cur.copy(
                    sql.SQL("COPY {} ({}) FROM STDIN").format(
                        sql.Identifier(dest),
                        sql.SQL(", ").join(map(sql.Identifier, cols)),
                    )
                ) as copy:
                    for row in rows:
                        copy.write_row(
                            [coerce(v, t) for v, t in zip(row, is_text)]
                        )
                normalize_catalog(con, dest, set(cols))
        finally:
            wb.close()


def ingest_data(con: psycopg.Connection, accdb_dir: str, drop_existing: bool,
                seen: dict[str, str]) -> None:
    """Load every non-rollup data table out of each year's .accdb."""
    banner(f"Ingesting data tables from {accdb_dir}")
    dbs = sorted(f for f in os.listdir(accdb_dir) if f.lower().endswith(".accdb"))
    if not dbs:
        raise SystemExit(f"No .accdb files in {accdb_dir}. Run --mode download first.")
    print(f"    {len(dbs)} Access DB(s) found", flush=True)
    for db in tqdm(dbs, desc="years"):
        accdb = os.path.join(accdb_dir, db)
        tables = subprocess.run(
            ["mdb-tables", "-1", accdb],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        # `-1` prints one name per line precisely because names can contain spaces.
        tables = [t.strip() for t in tables if t.strip()]

        for table in tqdm(tables, desc=db, leave=False):
            if table.upper().startswith(EXCLUDED_PREFIXES):
                continue
            # The accdb's own catalogs are superseded by the workbook sheets.
            if META_RE.match(table):
                continue

            dest = table.lower()  # so clients can write unquoted SQL
            if dest in seen:
                raise SystemExit(
                    f"Duplicate table name {dest!r} in {accdb} "
                    f"(first seen in {seen[dest]}). Resolve naming before ingest."
                )
            seen[dest] = accdb

            cols = get_schema(accdb, table)
            if drop_existing:
                con.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(dest))
                )
            con.execute(
                sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(dest), column_defs(cols)
                )
            )

            # Stream mdb-export into COPY as raw bytes: large tables are never
            # held in memory, and decoding would mangle non-ASCII.
            with subprocess.Popen(
                ["mdb-export", "-D", DATE_FMT, "-q", '"', "-X", '"', accdb, table],
                stdout=subprocess.PIPE,
            ) as export:
                with (
                    con.cursor() as cur,
                    cur.copy(
                        sql.SQL(
                            "COPY {} FROM STDIN WITH (FORMAT csv, HEADER true, "
                            "NULL '', QUOTE '\"', ESCAPE '\"')"
                        ).format(sql.Identifier(dest))
                    ) as copy,
                ):
                    while chunk := export.stdout.read(1 << 20):
                        copy.write(chunk)
                    if export.wait() != 0:
                        raise SystemExit(
                            f"mdb-export failed for {table} in {accdb} "
                            f"(exit {export.returncode}); see stderr above."
                        )


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
        description="Download IPEDS releases and ingest them into Postgres."
    )
    p.add_argument(
        "--mode",
        choices=("download", "ingest", "all"),
        default="all",
        help=(
            "download: fetch the release zips and extract their .accdb files."
            " ingest: load what is already on disk into Postgres."
            " all: both, in that order (default)."
        ),
    )
    p.add_argument(
        "--years",
        type=parse_years,
        # Runs to the current year so new releases need no edit here;
        # unpublished years 404 and are reported, not failed.
        default=f"{FIRST_IPEDS_YEAR}-{date.today().year}",
        help=(
            "Years to download, as '2019' or '2004-2023'"
            f" (default: {FIRST_IPEDS_YEAR} through the current year)."
            " Ignored by --mode ingest, which loads whatever is on disk."
        ),
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download release zips that are already present.",
    )
    p.add_argument(
        "--accdb-dir",
        default=ACCDB_PATH,
        help="Directory holding the IPEDS .accdb files (data tables).",
    )
    p.add_argument(
        "--zips-dir",
        default=ZIPS_PATH,
        help=(
            "Directory holding the IPEDS release zips, each carrying the"
            " IPEDS<YYYY><YY+1>Tablesdoc.xlsx the metadata catalogs are read from."
        ),
    )
    p.add_argument(
        "--database-url",
        default=IPEDS_DATABASE_URL,
        help="Postgres connection URL (defaults to $IPEDS_DATABASE_URL; there is no built-in default).",
    )
    p.add_argument(
        "--drop-existing",
        action="store_true",
        help=(
            "Drop each destination table before creating it, instead of refusing"
            " to run when a previous ingest is present. Only drops tables this"
            " run recreates; others must be dropped by hand."
        ),
    )
    return p.parse_args()


def run_ingest(args: argparse.Namespace) -> None:
    """Load the on-disk zips and .accdb files into Postgres."""
    if not args.database_url:
        raise SystemExit(
            "IPEDS_DATABASE_URL is not set. Add it to a .env file (see .env.example) "
            "or pass --database-url."
        )
    # The DB must already exist; fail loud with a hint if unreachable.
    try:
        con = psycopg.connect(args.database_url)
    except psycopg.OperationalError as e:
        raise SystemExit(
            f"Could not connect to the database: {e}\n"
            "Check the URL, and create the database first if it doesn't exist "
            "(e.g. `createdb <dbname>`)."
        ) from e
    con.autocommit = True
    # metadata tables carry '_meta'; their presence marks an existing ingest.
    existing = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name LIKE '%!_meta' ESCAPE '!'"
    ).fetchone()
    if existing is not None and existing[0] and not args.drop_existing:
        raise SystemExit(
            f"Database already has {existing[0]} *_meta tables; drop them first to "
            "re-ingest, or pass --drop-existing to overwrite them as they are rebuilt."
        )

    seen: dict[str, str] = {}  # table -> source file, to catch name collisions
    # Metadata first because it is quick: a malformed workbook then fails the
    # run before the multi-hour accdb load, not after it.
    ingest_metadata(con, args.zips_dir, args.drop_existing, seen)
    ingest_data(con, args.accdb_dir, args.drop_existing, seen)
    con.close()

    database = args.database_url.rsplit("/", 1)[-1]
    print(f"\nDone. Ingested {len(seen)} tables into {database}.", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode in ("download", "all"):
        download_releases(
            args.years, args.zips_dir, args.accdb_dir, args.force_download
        )
    if args.mode in ("ingest", "all"):
        run_ingest(args)


if __name__ == "__main__":
    main()
