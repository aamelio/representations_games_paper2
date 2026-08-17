#!/usr/bin/env python3
"""Download questionnaires (.qsf) and response data (.csv) from Qualtrics.

For every survey in the account whose name starts with NAME_PREFIX, this
script downloads into a subfolder <survey_name>_<survey_id>/ next to this
script:

    questionnaire.qsf       the survey definition (importable into Qualtrics)
    responses_numeric.csv   responses with numeric choice codes (e.g. 5)
    responses_labels.csv    responses with choice text (e.g. "Strongly agree")

Only COMPLETED responses are kept (rows with Finished = 1/True). Recorded
but unfinished responses (e.g. partials auto-closed when the survey ended)
are dropped; the script reports how many per survey. The full raw export is
always re-downloadable from Qualtrics, so nothing is lost by filtering here.

Re-running the script overwrites existing files (the surveys are closed, so
the data no longer changes).

SETUP (one-time)
----------------
1. Generate an API token: Qualtrics > Account Settings > Qualtrics IDs.
2. Store it in a credentials file the script can read (never commit this
   file or place it in Dropbox):

       printf 'QUALTRICS_API_TOKEN=PASTE_TOKEN_HERE\n' > ~/.qualtrics_credentials
       chmod 600 ~/.qualtrics_credentials

   Alternatively, export QUALTRICS_API_TOKEN as an environment variable;
   the environment variable takes precedence over the file.

RUN
---
    python3 download_qualtrics.py

Requires Python 3.8+ and the `requests` package (pip install requests).
API reference: https://api.qualtrics.com/
"""

import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATACENTER = "fra1"                       # from Account Settings > Qualtrics IDs
NAME_PREFIX = "main_collection_"          # download surveys whose name starts with this
OUTPUT_DIR = Path(__file__).resolve().parent   # subfolders are created next to this script
CREDENTIALS_FILE = Path.home() / ".qualtrics_credentials"

BASE_URL = f"https://{DATACENTER}.qualtrics.com/API/v3"
POLL_SECONDS = 2                          # how often to check an export's progress
EXPORT_TIMEOUT_SECONDS = 600              # give up on a single export after 10 minutes


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
def get_token() -> str:
    """Read the API token from the environment or the credentials file."""
    token = os.environ.get("QUALTRICS_API_TOKEN", "").strip()
    if token:
        return token
    if CREDENTIALS_FILE.exists():
        for line in CREDENTIALS_FILE.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "QUALTRICS_API_TOKEN" and value.strip():
                return value.strip()
    sys.exit(
        "No API token found. Either export QUALTRICS_API_TOKEN or create "
        f"{CREDENTIALS_FILE} containing a line 'QUALTRICS_API_TOKEN=...' "
        "(see the docstring at the top of this script)."
    )


def checked(response: requests.Response) -> dict:
    """Return the parsed JSON body, or exit with Qualtrics' error message."""
    if response.ok:
        return response.json()
    try:
        message = response.json()["meta"]["error"]["errorMessage"]
    except Exception:
        message = response.text[:500]
    sys.exit(f"Qualtrics API error ({response.status_code}) at {response.url}: {message}")


# --------------------------------------------------------------------------
# Survey list
# --------------------------------------------------------------------------
def list_all_surveys(session: requests.Session) -> list:
    """Return [{'id': ..., 'name': ...}, ...] for every survey in the account.

    The /surveys endpoint is paginated: each page returns up to 100 surveys
    and a 'nextPage' URL (null on the last page).
    """
    surveys = []
    url = f"{BASE_URL}/surveys"
    while url:
        result = checked(session.get(url, timeout=60))["result"]
        surveys.extend(result["elements"])
        url = result.get("nextPage")
    return surveys


# --------------------------------------------------------------------------
# Questionnaire (.qsf)
# --------------------------------------------------------------------------
def download_qsf(session: requests.Session, survey_id: str, dest: Path) -> None:
    """Save the survey definition in QSF format (same as the manual
    'Export Survey' in the Qualtrics editor; can be re-imported)."""
    response = session.get(
        f"{BASE_URL}/survey-definitions/{survey_id}",
        params={"format": "qsf"},
        timeout=60,
    )
    body = checked(response)
    # Depending on API version the QSF may arrive wrapped in {'result': ...};
    # a bare QSF is recognizable by its top-level 'SurveyEntry' key.
    qsf = body if "SurveyEntry" in body else body.get("result", body)
    dest.write_text(json.dumps(qsf, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Responses (.csv)
# --------------------------------------------------------------------------
def export_responses(session: requests.Session, survey_id: str,
                     use_labels: bool, dest: Path) -> tuple:
    """Export responses as CSV and save only completed rows to `dest`.

    The response export is asynchronous, in three steps:
      1. POST  .../export-responses            -> progressId
      2. GET   .../export-responses/{progress} -> poll until status 'complete'
      3. GET   .../export-responses/{file}/file -> zip containing one CSV

    use_labels=False -> numeric choice codes; True -> choice text.
    Returns (n_completed_kept, n_unfinished_dropped).
    """
    start = checked(session.post(
        f"{BASE_URL}/surveys/{survey_id}/export-responses",
        json={"format": "csv", "useLabels": use_labels, "compress": True},
        timeout=60,
    ))
    progress_id = start["result"]["progressId"]

    deadline = time.monotonic() + EXPORT_TIMEOUT_SECONDS
    while True:
        progress = checked(session.get(
            f"{BASE_URL}/surveys/{survey_id}/export-responses/{progress_id}",
            timeout=60,
        ))["result"]
        if progress["status"] == "complete":
            file_id = progress["fileId"]
            break
        if progress["status"] == "failed":
            sys.exit(f"Export failed for survey {survey_id}: {progress}")
        if time.monotonic() > deadline:
            sys.exit(f"Export timed out for survey {survey_id}.")
        time.sleep(POLL_SECONDS)

    download = session.get(
        f"{BASE_URL}/surveys/{survey_id}/export-responses/{file_id}/file",
        timeout=300,
    )
    download.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        raw_csv = archive.read(archive.namelist()[0])   # zip holds a single CSV

    return write_completed_only(raw_csv, dest)


def write_completed_only(raw_csv: bytes, dest: Path) -> tuple:
    """Write `raw_csv` to `dest`, keeping only completed responses.

    Qualtrics CSV exports have THREE header rows (variable names, question
    wording, ImportId JSON); these are always preserved. Data rows are kept
    when the 'Finished' column is 1 (numeric export) or True (labels export).
    Returns (n_kept, n_dropped).
    """
    rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    header, data = rows[:3], rows[3:]

    try:
        finished_col = header[0].index("Finished")
        kept = [r for r in data if r[finished_col] in {"1", "True", "TRUE", "true"}]
    except (ValueError, IndexError):
        print("    WARNING: no 'Finished' column found; keeping all rows.")
        kept = data

    # utf-8-sig preserves the byte-order mark Qualtrics ships, so the file
    # opens correctly in Excel and reads identically in R/Stata/pandas.
    with dest.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(header + kept)
    return len(kept), len(data) - len(kept)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """Make a survey name filesystem-safe (spaces/punctuation -> underscore)."""
    return re.sub(r"[^\w.-]+", "_", name).strip("_")


def main() -> None:
    session = requests.Session()
    session.headers["X-API-TOKEN"] = get_token()

    all_surveys = list_all_surveys(session)
    matched = sorted(
        (s for s in all_surveys if s["name"].startswith(NAME_PREFIX)),
        key=lambda s: s["name"],
    )
    print(f"Account has {len(all_surveys)} surveys; "
          f"{len(matched)} match prefix '{NAME_PREFIX}'.")
    if not matched:
        print("Survey names found:")
        for s in sorted(all_surveys, key=lambda s: s["name"]):
            print(f"  {s['name']}")
        sys.exit(1)

    for survey in matched:
        folder = OUTPUT_DIR / f"{safe_name(survey['name'])}_{survey['id']}"
        folder.mkdir(exist_ok=True)
        print(f"\n{survey['name']} ({survey['id']})")

        download_qsf(session, survey["id"], folder / "questionnaire.qsf")
        print("    questionnaire.qsf saved")

        for use_labels, filename in [(False, "responses_numeric.csv"),
                                     (True, "responses_labels.csv")]:
            kept, dropped = export_responses(
                session, survey["id"], use_labels, folder / filename)
            print(f"    {filename}: {kept} completed responses"
                  + (f" ({dropped} unfinished dropped)" if dropped else ""))

    print(f"\nDone. Files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
