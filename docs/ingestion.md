# Ingesting and post-processing IPEDS

`db_setup/ingest.py` loads the survey data into `idea_ipeds`. Everything after
it is optional: the two `postprocess` scripts compress the descriptions the
agent reads, and the app works fine without any of it.

---

## Ingest

`ingest.py` downloads NCES release zips into `data/zips/`, extracts the Access
files into `data/accdb/`, and loads them into `idea_ipeds`. It shells out to
`mdb-export`, so `mdbtools` has to be installed (`brew install mdbtools`).

```console
python db_setup/ingest.py --years 2023
```

Omit `--years` to pull every year from 2004 on — that is ~728 data tables and
takes hours. Start with one year to get running. `--mode download` and
`--mode ingest` split the two halves if you want to fetch once and reload later.

`db_setup/fetch_db.sh` downloads the same zips from a shell, one year or
`--range first last`, and keeps only the `.accdb`. It predates `ingest.py` and
is not needed for a normal run — `--mode download` covers the same ground.

---

## Table descriptions — `postprocess.py`

`db_setup/postprocess.py` compresses the table descriptions the agent reads. It
works one `(year, survey)` group at a time rather than one table at a time,
because the redundancy worth removing is redundancy *within* a survey: all six
Fall Enrollment tables repeat the same paragraph about semester calendars and
the October 15 reporting date, and only a clause or two tells them apart. Each
group yields a survey summary, written to a new `surveys<YY>_meta` table, plus a
per-table summary holding only the discriminating detail, written to
`tables<YY>_meta.description_summary`. The agent's
`get_table_descriptions_by_year` already prefers that column when it exists, so
nothing else needs changing.

It uses Azure OpenAI, so it needs `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`) and
`AZURE_OPENAI_DEPLOYMENT_NAME`. Sample before committing to a full run:

```bash
# One group, printed, nothing written.
python db_setup/postprocess.py --years 2023 --survey "Fall Enrollment" --dry-run
python db_setup/postprocess.py --years 2023        # write one year
python db_setup/postprocess.py                     # all years, ~190 calls
```

`--limit N` caps the number of groups. Re-running is safe: a group whose survey
row exists and whose tables are all summarized is skipped, so an interrupted run
resumes.

Every model call is also banked to CSV under `data/summaries/` — one
`surveys<YY>_meta.csv` and one `tables<YY>_meta.csv` per year, rewritten as a
complete snapshot each run so a partial or repeated run leaves one row per key.
That directory is committed (the one exception to `data/` being gitignored), so
you never pay for these summaries twice: `--from-csv` reads the banked files and
writes straight to Postgres with no model call at all, and so needs no Azure
credentials. A fresh clone can restore every year this way.

```bash
python db_setup/postprocess.py --from-csv              # restore every banked year
python db_setup/postprocess.py --years 2023 --from-csv # or just one
```

It fails loudly rather than half-loading: a group missing from the CSVs, or a
CSV missing one of the group's tables, stops the run naming what to re-generate.
`--csv-dir` moves the directory and `--no-csv` turns the banking off.

---

## Column descriptions — `postprocess_columns.py`

`db_setup/postprocess_columns.py` does the same job one level down, for the
*column* descriptions in `vartable<YY>_meta`. Those are the bigger problem: one
call to `get_column_descriptions_of_table(2023, 'sfa2223_p1')` hands the agent
555,065 characters — about 139,000 tokens for a single tool result — most of it
net-price arithmetic and a glossary repeated on every column.

Two things about the corpus set the design. It repeats across years: deduping on
the description with year tokens masked (`2022-23` → `YYYY`) takes 33,871 rows
down to 3,714 distinct texts, because a variable's wording is revised
occasionally rather than rewritten annually. So the unit of work is a distinct
text, not a row — summarize once, write the result to every row that shares it,
in every year. And the weight is concentrated: descriptions of 800+ characters
are 32% of rows but 73% of the characters, so `--min-chars` (default 800) buys
most of the payload back for a third of the calls. Shorter descriptions are left
alone, `longdescription_summary` stays NULL for them, and the reader coalesces
to the raw text.

Because one summary is shared across every year that uses the text, the
summaries are written year-generically; the concrete year reaches the agent
through the variable's title and the table name instead.
`get_column_descriptions_of_table` prefers the column when it exists.

```bash
# The five highest-reuse descriptions, printed, nothing written.
python db_setup/postprocess_columns.py --limit 5 --dry-run
python db_setup/postprocess_columns.py --table sfa2223_p1   # one table's descriptions
python db_setup/postprocess_columns.py                      # all years, ~1,400 calls
```

The work list is ordered by how many rows share each description, so `--limit`
samples the ones that matter and an interrupted run has already bought the
biggest reduction. `--concurrency` (default 4) parallelizes the model calls;
database writes stay single-threaded. Re-running is safe: a description whose
rows all have a summary is skipped.

Banking works the same way as `postprocess.py`, into one committed
`data/summaries/columns.csv`, flushed every 50 calls so a crash late in a long
run doesn't discard the tokens already spent. The masked text itself is too big
to store, so rows are keyed by its SHA-256 along with a `norm_version` — bumped
whenever the normalization changes, so a change surfaces as a version mismatch
rather than as a silent flood of keys that suddenly look unbanked. `--from-csv`
validates every key up front and refuses to write anything if one is missing.

---

## Re-ingesting

One wrinkle if you later re-ingest: `ingest.py` decides a previous ingest exists
by counting tables matching `%_meta`, and `surveys<YY>_meta` matches that
pattern. `--drop-existing` won't drop it, because `ingest.py` only drops tables
it recreates — drop the `surveys*_meta` tables by hand if you want a clean
slate.
