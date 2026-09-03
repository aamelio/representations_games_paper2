"""Resumable GPT-5.6 classification of within-subject HP descriptions.

The exact classification prompt is stored under
within_subject/prompts/classification_prompt.txt. Exact, consistently labelled
texts from the finalized between-subject GPT-5.6 run are inherited; every other
unique nonempty text is sent as its own Batch API request.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WITHIN = ROOT / "within_subject"
INPUT = WITHIN / "input"
WORK = WITHIN / "work"
DATA = WITHIN / "data"
PROMPT_PATH = WITHIN / "prompts" / "classification_prompt.txt"
UNIQUE_PATH = INPUT / "within_hp_unique_texts.csv"
PANEL_PATH = INPUT / "within_hp_panel_unclassified.csv"
PROGRESS_PATH = WORK / "classification_progress.csv"
REQUESTS_PATH = WORK / "classification_requests.jsonl"
STATE_PATH = WORK / "batch_state.json"
OUTPUT_PATH = WORK / "batch_output.jsonl"
FINAL_PATH = DATA / "within_hp_panel_classified.csv"
BETWEEN_CLASSIFIED = (
    ROOT.parent
    / "rounds4_9"
    / "hp_similarity"
    / "data"
    / "hp_responses_classified_and_rated.csv"
)

MODEL = "gpt-5.6-sol"
CATEGORY_LABELS = {
    0: "No clear justification",
    1: "Moral",
    2: "Mutual Benefit / Cooperation",
    3: "Self-interest",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_key(value: object) -> str:
    return clean_text(value).lower()


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, path)


def inherited_lookup() -> dict[str, int]:
    prior = pd.read_csv(BETWEEN_CLASSIFIED)
    prior["memory_key"] = prior["memory"].map(normalized_key)
    counts = prior.groupby("memory_key", observed=True)["category_num"].nunique()
    consistent = set(counts[counts == 1].index)
    lookup = (
        prior[prior["memory_key"].isin(consistent)]
        .drop_duplicates("memory_key")
        .set_index("memory_key")["category_num"]
        .astype(int)
        .to_dict()
    )
    return lookup


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    unique = pd.read_csv(UNIQUE_PATH)
    inherited = inherited_lookup()
    if PROGRESS_PATH.exists():
        existing = pd.read_csv(PROGRESS_PATH)
        completed = (
            existing.dropna(subset=["category_num"])
            .set_index("classification_id")
            [["category_num", "classification_origin"]]
        )
    else:
        completed = pd.DataFrame(
            columns=["category_num", "classification_origin"]
        )
    progress = unique.copy()
    progress["category_num"] = progress["memory_key"].map(inherited)
    progress["classification_origin"] = progress["category_num"].map(
        lambda value: "inherited_gpt_5_6_between"
        if pd.notna(value)
        else ""
    )
    for classification_id, row in completed.iterrows():
        mask = progress["classification_id"].eq(classification_id)
        if mask.any():
            progress.loc[mask, "category_num"] = int(row["category_num"])
            progress.loc[mask, "classification_origin"] = row[
                "classification_origin"
            ]
    progress["category_num"] = pd.to_numeric(
        progress["category_num"], errors="coerce"
    ).astype("Int64")
    progress["category"] = progress["category_num"].map(CATEGORY_LABELS)
    atomic_write_csv(progress, PROGRESS_PATH)
    build_requests(progress)
    show_status()


def build_requests(progress: pd.DataFrame) -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    pending = progress[progress["category_num"].isna()]
    with REQUESTS_PATH.open("w", encoding="utf-8") as handle:
        for row in pending.itertuples(index=False):
            request = {
                "custom_id": row.classification_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "instructions": prompt,
                    "input": row.memory,
                    "max_output_tokens": 16,
                    "reasoning": {"effort": "low"},
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")


def require_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available in this environment.")
    from openai import OpenAI

    return OpenAI()


def submit() -> None:
    prepare()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("status") not in {
            "failed",
            "expired",
            "cancelled",
            "completed",
        }:
            raise RuntimeError(
                f"Batch {state['batch_id']} is already {state['status']}."
            )
    client = require_client()
    with REQUESTS_PATH.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"analysis": "within_subject_hp_classification"},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model": MODEL,
        "status": batch.status,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
    }
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2))


def update_remote_status(client, state: dict):
    batch = client.batches.retrieve(state["batch_id"])
    state["status"] = batch.status
    state["request_counts"] = {
        "total": batch.request_counts.total,
        "completed": batch.request_counts.completed,
        "failed": batch.request_counts.failed,
    }
    state["output_file_id"] = batch.output_file_id
    state["error_file_id"] = batch.error_file_id
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return batch, state


def extract_output_text(body: dict) -> str:
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(str(content.get("text", "")))
    return "".join(texts).strip()


def parse_batch_output(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        custom_id = record["custom_id"]
        response = record.get("response")
        if not response or response.get("status_code") != 200:
            continue
        text = extract_output_text(response["body"])
        match = re.fullmatch(r"[0-3]", text)
        if match:
            rows.append({
                "classification_id": custom_id,
                "category_num": int(text),
                "classification_origin": f"{MODEL}_batch",
            })
    return pd.DataFrame(rows)


def merge_results(results: pd.DataFrame) -> None:
    progress = pd.read_csv(PROGRESS_PATH)
    if not results.empty:
        results = results.drop_duplicates("classification_id", keep="last")
        lookup = results.set_index("classification_id")
        for classification_id, row in lookup.iterrows():
            mask = progress["classification_id"].eq(classification_id)
            progress.loc[mask, "category_num"] = int(row["category_num"])
            progress.loc[mask, "classification_origin"] = row[
                "classification_origin"
            ]
    progress["category_num"] = pd.to_numeric(
        progress["category_num"], errors="coerce"
    ).astype("Int64")
    progress["category"] = progress["category_num"].map(CATEGORY_LABELS)
    atomic_write_csv(progress, PROGRESS_PATH)
    build_requests(progress)


def collect() -> None:
    client = require_client()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    batch, state = update_remote_status(client, state)
    if batch.status != "completed":
        print(json.dumps(state, indent=2))
        return
    if not batch.output_file_id:
        raise RuntimeError("Completed batch has no output file.")
    content = client.files.content(batch.output_file_id)
    OUTPUT_PATH.write_bytes(content.content)
    results = parse_batch_output(OUTPUT_PATH)
    merge_results(results)
    finalize()


def import_results(path: Path) -> None:
    supplied = pd.read_csv(path)
    required = {"classification_id", "category_num"}
    if not required.issubset(supplied.columns):
        raise ValueError(f"Manual results require columns {sorted(required)}.")
    supplied["category_num"] = pd.to_numeric(
        supplied["category_num"], errors="raise"
    ).astype(int)
    if not supplied["category_num"].isin(CATEGORY_LABELS).all():
        raise ValueError("Manual result labels must be integers from 0 to 3.")
    supplied["classification_origin"] = supplied.get(
        "classification_origin", "gpt_5_6_manual_packet"
    )
    merge_results(
        supplied[
            ["classification_id", "category_num", "classification_origin"]
        ]
    )
    finalize()


def finalize() -> None:
    progress = pd.read_csv(PROGRESS_PATH)
    unresolved = progress["category_num"].isna().sum()
    if unresolved:
        print(f"Classification incomplete: {unresolved} unique texts remain.")
        return
    panel = pd.read_csv(PANEL_PATH)
    lookup = progress[
        [
            "memory_key",
            "category_num",
            "category",
            "classification_origin",
        ]
    ]
    panel = panel.merge(
        lookup, on="memory_key", how="left", validate="many_to_one"
    )
    blank = panel["memory_key"].fillna("").eq("")
    panel.loc[blank, "category_num"] = 0
    panel.loc[blank, "category"] = CATEGORY_LABELS[0]
    panel.loc[blank, "classification_origin"] = "empty_text"
    if panel["category_num"].isna().any():
        raise ValueError("Final panel contains unresolved nonempty texts.")
    panel["category_num"] = panel["category_num"].astype(int)
    atomic_write_csv(panel, FINAL_PATH)
    print(f"Finalized {len(panel)} HP rows at {FINAL_PATH}.")


def show_status() -> None:
    if not PROGRESS_PATH.exists():
        print("Classification progress has not been prepared.")
        return
    progress = pd.read_csv(PROGRESS_PATH)
    complete = progress["category_num"].notna().sum()
    total = len(progress)
    print(f"Unique nonempty texts: {total}")
    print(f"Classified: {complete}")
    print(f"Remaining: {total - complete}")
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print(f"Remote batch: {state.get('batch_id')} ({state.get('status')})")
    print(f"Final row-level panel: {'present' if FINAL_PATH.exists() else 'missing'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["prepare", "submit", "collect", "status", "finalize", "import"],
    )
    parser.add_argument("--results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "submit":
        submit()
    elif args.action == "collect":
        collect()
    elif args.action == "status":
        show_status()
    elif args.action == "finalize":
        finalize()
    else:
        if args.results is None:
            raise ValueError("--results is required for the import action.")
        import_results(args.results)


if __name__ == "__main__":
    main()
