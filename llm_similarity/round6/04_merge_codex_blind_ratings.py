#!/usr/bin/env python3
"""Validate and combine the three blinded in-task Codex rating passes."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "round6-neutral-v1"
PARTS = [HERE / f"codex_blind_ratings_rep{replicate}.csv" for replicate in range(1, 4)]
INPUT_FILES = [
    HERE / "anonymized_vignettes.csv",
    HERE / "anonymized_contexts.csv",
    HERE / "anonymization_map.json",
]
OUTPUT = HERE / "similarity_ratings.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint() -> str:
    digest = hashlib.sha256(PROMPT_VERSION.encode("utf-8"))
    for path in INPUT_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_part(
    rows: list[dict[str, str]], expected_contexts: set[str], expected_vignettes: set[str], path: Path
) -> None:
    if list(rows[0]) != ["context_id", "vignette_id", "rating"]:
        raise ValueError(f"{path.name}: unexpected columns {list(rows[0])}")
    expected_pairs = {(context, vignette) for context in expected_contexts for vignette in expected_vignettes}
    observed_pairs = {(row["context_id"], row["vignette_id"]) for row in rows}
    if len(rows) != 360 or len(observed_pairs) != 360 or observed_pairs != expected_pairs:
        raise ValueError(
            f"{path.name}: expected exactly all 360 context-vignette pairs; "
            f"rows={len(rows)}, unique_pairs={len(observed_pairs)}, "
            f"missing={len(expected_pairs - observed_pairs)}, extra={len(observed_pairs - expected_pairs)}"
        )
    bad = []
    for row in rows:
        try:
            value = float(row["rating"])
        except ValueError:
            bad.append((row["context_id"], row["vignette_id"], row["rating"]))
            continue
        if not 0 <= value <= 100:
            bad.append((row["context_id"], row["vignette_id"], value))
    if bad:
        raise ValueError(f"{path.name}: invalid ratings, first examples={bad[:5]}")


def main() -> None:
    expected_contexts = {row["t_id"] for row in read_csv(HERE / "anonymized_contexts.csv")}
    expected_vignettes = {row["v_id"] for row in read_csv(HERE / "anonymized_vignettes.csv")}
    if len(expected_contexts) != 12 or len(expected_vignettes) != 30:
        raise ValueError("The blinded input files do not contain the expected 12 contexts and 30 vignettes")

    input_fingerprint = fingerprint()
    timestamp = datetime.now(timezone.utc).isoformat()
    output_rows: list[dict] = []
    for replicate, path in enumerate(PARTS, start=1):
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_csv(path)
        validate_part(rows, expected_contexts, expected_vignettes, path)
        for row in rows:
            output_rows.append(
                {
                    "provider": "codex",
                    "model": "codex-self-rating",
                    "replicate": replicate,
                    "context_id": row["context_id"],
                    "vignette_id": row["vignette_id"],
                    "presentation_label": row["vignette_id"],
                    "presentation_order": int(row["vignette_id"][1:]),
                    "rating": float(row["rating"]),
                    "prompt_version": PROMPT_VERSION,
                    "attempts": 1,
                    "timestamp_utc": timestamp,
                    "input_fingerprint": input_fingerprint,
                }
            )

    fields = list(output_rows[0])
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(OUTPUT)

    protocol = {
        "description": "Three blinded Codex rating passes generated within the Codex task.",
        "important_limitation": (
            "These are repeated judgments from the same Codex model family, not independent external "
            "models or human raters. They are suitable for an exploratory estimate, not cross-model validation."
        ),
        "prompt_version": PROMPT_VERSION,
        "input_fingerprint": input_fingerprint,
        "replicates": 3,
        "contexts_per_replicate": 12,
        "vignettes_per_context": 30,
        "rows": len(output_rows),
    }
    (HERE / "codex_rating_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output_rows)} validated ratings to {OUTPUT}")


if __name__ == "__main__":
    main()
