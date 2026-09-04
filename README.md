# 🎓 IDEA: IPEDS Data Exploration Assistant

An AI-assisted analytics platform for exploring U.S. higher-education datasets (IPEDS) using natural language.

A single LLM agent with read-only SQL access answers questions over 20 years of IPEDS survey tables (2004-05 through 2023-24). The agent discovers the schema through tools rather than being handed a schema dump. See [docs/design.md](docs/design.md) for architecture and [docs/interrupts.md](docs/interrupts.md) for how a turn pauses mid-run when the agent asks a clarifying question or a saved preference needs approval.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL 14+ (two databases: `idea_main`, `idea_ipeds`) — or Docker, which ships one (`postgres:18`) and can run the app itself
- `mdbtools` — only for IPEDS ingestion, which reads Access files (`brew install mdbtools`)
- An Anthropic API key

No frontend build step: `frontend/` is htmx and plain JS, served as-is.

---

## 1. Install dependencies

```console
uv sync
source .venv/bin/activate
```

## 2. Environment variables

```console
cp .env.example .env
```

Then fill in `ANTHROPIC_API_KEY` and `JWT_SECRET`. Generate the latter with `python -c "import secrets; print(secrets.token_urlsafe(48))"` — it has no default, and the app refuses to start without it. The defaults in the file cover everything else for a local run.

| Variable | Required | Notes |
|---|---|---|
| `JWT_SECRET` | **yes** | Signs session JWTs and the session cookie. No default. |
| `MAIN_DATABASE_URL` | yes | App tables (users, sessions, messages, checkpoints) |
| `IPEDS_DATABASE_URL` | yes | Survey data, read-only at runtime |
| `ANTHROPIC_API_KEY` | yes | Read by `langchain-anthropic` |
| `AGENT_MODEL` | no | Defaults to `anthropic:claude-sonnet-4-5` |
| `CHECKPOINT_DATABASE_URL` | no | Defaults to `MAIN_DATABASE_URL` |
| `CONTEXT_WINDOW` | no | Defaults to `200000`. Denominator for the context meter under the composer. |
| `CLEAR_TOOL_USES_TRIGGER` | no | Defaults to `30000`. Token count at which old tool results start being cleared. |
| `RECURSION_LIMIT` | no | Defaults to `50` graph steps (~24 tool calls) per turn, so a looping agent can't bill indefinitely. |
| `DEV_AUTH_BYPASS` | no | Defaults to `0`. See below. |
| `COOKIE_SECURE` | no | Defaults to `1`. Set to `0` for local http. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REDIRECT_URI` | prod | Google OAuth |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_DEPLOYMENT_NAME` | postprocess | Only for `db_setup/postprocess*.py`. See [docs/ingestion.md](docs/ingestion.md). |
| `LANGSMITH_API_KEY` | eval | Only for `eval/run_eval.py`, which scores through LangSmith. |

## 3. Create the databases

The app creates its own tables on startup (`create_all` + the LangGraph checkpointer), so you only need the two empty databases. Use whatever Postgres you already have — any role that can create databases works:

```console
createdb idea_main
createdb idea_ipeds
```

If you don't have a Postgres, `docker-compose.yml` ships one — `postgres:18` as container `idea-pg`, user `postgres`, password `idea`, published on 5432. Start it on its own:

```console
docker compose up -d db
docker exec idea-pg createdb -U postgres idea_ipeds
```

It creates `idea_main` itself from `POSTGRES_DB`; `idea_ipeds` is the one command above. Either way, make sure the URLs in `.env` match the Postgres you just used — for the bundled one that is `postgresql://postgres:idea@localhost:5432/...`, which is what `.env.example` already has.

The named volume `pgdata` is what keeps the data. Without it Postgres writes into the container's own layer and a `docker rm idea-pg` destroys both databases — including a full IPEDS ingest that took hours. With it, you can remove and recreate the container freely; only `docker volume rm pgdata` erases the data.

## 4. Add yourself to the allowlist

Sign-in is allowlisted even with `DEV_AUTH_BYPASS=1`, so add your email to `ALLOWED_EMAILS` in `backend/app/config.py` or every login attempt returns 403. See [Authentication](#authentication-and-dev_auth_bypass) below.

## 5. Load the IPEDS data

`db_setup/ingest.py` downloads NCES release zips, extracts the Access files, and loads them into `idea_ipeds`. It needs `mdbtools`.

```console
python db_setup/ingest.py --years 2023
```

Omit `--years` to pull every year from 2004 on — that is ~728 data tables and takes hours. Start with one year to get running.

[docs/ingestion.md](docs/ingestion.md) covers the rest: the download/ingest split, and the optional `postprocess.py` / `postprocess_columns.py` passes that compress the table and column descriptions the agent reads. The app works fine without them.

Their output is already banked in `data/summaries/`, so you can apply it without an Azure account or a single model call:

```console
python db_setup/postprocess.py --from-csv
python db_setup/postprocess_columns.py --from-csv
```

## 6. Run

```console
uvicorn backend.app.main:app --reload --port 8000
```

Open http://localhost:8000.

Or with Docker Compose, which builds the app image and brings up the Postgres from step 3 alongside it:

```console
docker compose up --build
```

Two services: `app` on port 8000, and `db`. The app reads `MAIN_DATABASE_URL` and `IPEDS_DATABASE_URL` straight out of `.env` — compose fails with a message naming the variable if either is missing — so the host in those URLs has to be one the container can reach. `localhost` inside the container is the container, not your machine: use `db` for the bundled Postgres, or `host.docker.internal` for one running on your host. There is only one pair of URLs, so switch the host when you switch between `uvicorn` and compose.

---

## Evaluation

`eval/` scores the agent against ground truth scraped from the NCES [Trend Generator](https://nces.ed.gov/ipeds/trendgenerator/). The dataset is committed as `eval/dataset.jsonl`, so a run needs no scraping: `fetch_raw.py` → `build_dataset.py` → `paraphrase.py` only need re-running to rebuild it.

```console
python eval/run_eval.py --push-only          # upload the dataset, run nothing
python eval/run_eval.py --limit 10 --dry-run # see what would run
python eval/run_eval.py --limit 10           # run it — costs money
```

It runs the production prompt and tools from `backend/app/agent.py`, minus the two tools that wait on a human, plus an eval-only prompt suffix pinning the published IPEDS universe that no Trend Generator question states out loud. Scoring goes through LangSmith, so it needs `LANGSMITH_API_KEY`. `--model` takes a LangChain model id when it contains a colon (`anthropic:claude-sonnet-4-5`) and an Azure deployment name otherwise; the default is an Azure deployment. `--year`, `--variant`, and `--latest-only` narrow the set. [eval/README.md](eval/README.md) documents the dataset and the two Trend Generator quirks that shape it.

---

## Authentication and `DEV_AUTH_BYPASS`

In production the app signs in through Google OAuth and only accepts emails in `ALLOWED_EMAILS` in `backend/app/config.py`.

Locally you almost certainly don't want to set up a Google Cloud project. Set `DEV_AUTH_BYPASS=1` and the login page shows an email box instead: enter any email on the allowlist and you get the same session JWT, no Google round-trip. Add your address to `ALLOWED_EMAILS` first — the allowlist is still enforced.

The flag is off by default and gates the `/auth/dev-login` endpoint on its own, so a deployed environment that forgets to set it cannot be signed into this way. It also takes precedence over OAuth: with it on, `/auth/login` goes to the dev picker even if Google credentials are configured. Never set it in a deployed environment.

Pair it with `COOKIE_SECURE=0` locally — browsers drop `Secure` cookies over plain http, and login silently fails without it.

---

## Health check

```console
curl http://localhost:8000/health
```
