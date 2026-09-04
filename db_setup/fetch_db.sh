#!/usr/bin/env bash
#
# Download IPEDS Access DB zips and extract the .accdb file from each.
#
# IPEDS Access DBs are published per academic year at:
#   https://nces.ed.gov/ipeds/tablefiles/zipfiles/IPEDS_<YYYY>-<YY+1>_Final.zip
# Each zip holds one .accdb (e.g. IPEDS200405.accdb) plus other files we don't
# need. We keep only the .accdb and drop everything else.
#
# Usage:
#   ./fetch_db.sh <year>                              # one year, e.g. 2011
#   ./fetch_db.sh --range <first_year> <last_year>    # inclusive range
#   ./fetch_db.sh <year|--range ...> -o <dir>         # custom output dir

set -euo pipefail

FIRST_YEAR=""
LAST_YEAR=""
BASE_URL="https://nces.ed.gov/ipeds/tablefiles/zipfiles"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") (<year> | --range <first_year> <last_year>) [-o <out_dir>]

  <year>                    a single four-digit year to download
  --range <first> <last>    download an inclusive range of four-digit years
  -o, --out <dir>           output directory for .accdb files (default: <project_root>/data/accdb)
EOF
    exit 2
}

# --- parse args ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(dirname "$SCRIPT_DIR")/data/accdb"  # project_root/data/accdb
YEAR=""

# Process command-line arguments, consuming each one with shift.
while [[ $# -gt 0 ]]; do
    arg="$1"

    case "$arg" in
        --range)
            if [[ $# -lt 3 ]]; then
                echo "error: $arg needs first and last year arguments" >&2
                usage
            fi
            if [[ ! "$2" =~ ^[0-9]{4}$ || ! "$3" =~ ^[0-9]{4}$ ]]; then
                echo "error: --range years must each contain four digits" >&2
                usage
            fi
            if [[ -n "$FIRST_YEAR" ]]; then
                echo "error: multiple ranges given" >&2
                usage
            fi

            FIRST_YEAR="$2"
            LAST_YEAR="$3"
            shift 3
            ;;
        -o|--out)
            if [[ $# -lt 2 ]]; then
                echo "error: $arg needs a directory argument" >&2
                usage
            fi

            OUT="$2"
            shift 2
            ;;
        [0-9][0-9][0-9][0-9])
            if [[ -n "$YEAR" ]]; then
                echo "error: multiple years given" >&2
                usage
            fi

            YEAR="$arg"
            shift
            ;;
        *)
            echo "error: unrecognized argument: $arg" >&2
            usage
            ;;
    esac
done

# Require exactly one of <year> / --range.
if [[ -n "$YEAR" && -n "$FIRST_YEAR" ]]; then
    echo "error: give either a year or --range, not both" >&2
    usage
fi
if [[ -z "$YEAR" && -z "$FIRST_YEAR" ]]; then
    echo "error: provide a year or --range" >&2
    usage
fi
if [[ -n "$FIRST_YEAR" ]] && (( 10#${FIRST_YEAR} > 10#${LAST_YEAR} )); then
    echo "error: first year must not be later than last year" >&2
    usage
fi

# --- download one year -----------------------------------------------------
download_year() {
    local year="$1"
    local y1 url tmp
    y1=$(printf '%02d' $(( (year + 1) % 100 )))
    url="${BASE_URL}/IPEDS_${year}-${y1}_Final.zip"

    tmp=$(mktemp -d)

    echo "==> ${year}-${y1}: downloading"
    if ! curl -fSL --retry 3 -o "$tmp/ipeds.zip" "$url"; then
        echo "warning: failed to download ${url}" >&2
        rm -rf "$tmp"
        return 1
    fi

    # -j junk paths (flatten), -o overwrite; the *.accdb filter keeps only the
    # Access DB and leaves everything else in the zip behind. A corrupt zip or one
    # with no .accdb member is reported like a download failure, so a --range run
    # counts it and carries on instead of aborting under `set -e`.
    if ! unzip -o -j "$tmp/ipeds.zip" '*.accdb' -d "$OUT"; then
        echo "warning: failed to extract an .accdb from ${url}" >&2
        rm -rf "$tmp"
        return 1
    fi

    rm -rf "$tmp"
    echo "==> ${year}-${y1}: done"
}

# --- main ------------------------------------------------------------------
mkdir -p "$OUT"

if [[ -n "$FIRST_YEAR" ]]; then
    failures=0
    for year in $(seq "$FIRST_YEAR" "$LAST_YEAR"); do
        download_year "$year" || failures=$((failures + 1))
    done
    if (( failures > 0 )); then
        echo "finished with ${failures} failed year(s)" >&2
        exit 1
    fi
    echo "Downloaded years ${FIRST_YEAR} through ${LAST_YEAR} into ${OUT}/"
else
    download_year "$YEAR"
    echo "Downloaded into ${OUT}/"
fi
