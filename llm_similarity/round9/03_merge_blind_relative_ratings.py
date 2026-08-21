#!/usr/bin/env python3
"""Validate and combine the three blinded round-9 rating passes."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUTS = {rater: HERE / f"relative_agent{rater}_ratings.csv" for rater in (1, 2, 3)}
PACKETS = {rater: HERE / "relative_packets" / f"rater_{rater}_tasks.json" for rater in INPUTS}
REQUIRED_COLUMNS = ["rater", "replicate", "task", "context_id", "vignette_id", "points"]


def expected_pairs(packet_path: Path) -> set[tuple[str, str, str]]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    return {
        (task["stage"], task["context_id"], vignette["vignette_id"])
        for task in packet["tasks"] for vignette in task["vignettes"]
    }


def validate_one(path: Path, packet_path: Path, rater_number: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    if list(data.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {REQUIRED_COLUMNS}")
    if len(data) != 352:
        raise ValueError(f"{path.name}: expected 352 rows, found {len(data)}")
    if set(data["rater"]) != {f"relative-agent-{rater_number}"}:
        raise ValueError(f"{path.name}: unexpected rater label")
    if set(data["replicate"]) != {1}:
        raise ValueError(f"{path.name}: replicate must equal 1")
    data["points"] = pd.to_numeric(data["points"], errors="raise")
    if (data["points"] < 0).any() or not np.allclose(data["points"], np.round(data["points"])):
        raise ValueError(f"{path.name}: points must be nonnegative integers")
    data["points"] = data["points"].astype(int)
    key = ["task", "context_id", "vignette_id"]
    if data.duplicated(key).any():
        raise ValueError(f"{path.name}: duplicate task-context-vignette rows")
    observed = set(map(tuple, data[key].itertuples(index=False, name=None)))
    expected = expected_pairs(packet_path)
    if observed != expected:
        raise ValueError(
            f"{path.name}: candidate mismatch; missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    sums = data.groupby(["task", "context_id"])["points"].sum()
    if len(sums) != 24 or not (sums == 1000).all():
        raise ValueError(f"{path.name}: every one of the 24 tasks must sum to 1,000")
    return data


def main() -> None:
    frames = [validate_one(INPUTS[rater], PACKETS[rater], rater) for rater in sorted(INPUTS)]
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 1056:
        raise AssertionError("Expected 1,056 combined rating rows")
    combined.to_csv(HERE / "relative_ratings_3agents.csv", index=False)

    means = combined.groupby(["task", "context_id", "vignette_id"], as_index=False).agg(
        mean_points=("points", "mean"), sd_points=("points", "std"),
        min_points=("points", "min"), max_points=("points", "max"),
        n_raters=("rater", "nunique"),
    )
    if len(means) != 352 or not (means["n_raters"] == 3).all():
        raise AssertionError("Every candidate must have exactly three ratings")
    if not np.allclose(means.groupby(["task", "context_id"])["mean_points"].sum(), 1000):
        raise AssertionError("Averaged allocations must sum to 1,000")
    means.to_csv(HERE / "relative_ratings_3agent_means.csv", index=False)

    agreement_rows = []
    for (task, context_id), group in combined.groupby(["task", "context_id"]):
        wide = group.pivot(index="vignette_id", columns="rater", values="points")
        for rater_a, rater_b in combinations(sorted(wide.columns), 2):
            first, second = wide[rater_a], wide[rater_b]
            agreement_rows.append({
                "task": task, "context_id": context_id, "rater_a": rater_a, "rater_b": rater_b,
                "pearson": first.corr(second),
                "spearman": first.rank(method="average").corr(second.rank(method="average")),
                "mean_absolute_point_difference": (first - second).abs().mean(),
                "total_variation_distance": 0.5 * (first - second).abs().sum() / 1000,
            })
    agreement = pd.DataFrame(agreement_rows)
    if len(agreement) != 72:
        raise AssertionError("Expected three pairwise comparisons for each of 24 tasks")
    agreement.to_csv(HERE / "relative_rater_agreement.csv", index=False)

    protocol = {
        "description": "Direct 1,000-point relative-similarity ratings after neutralizing incidental affect/own-payoff wording in DG-C, averaged across three separately tasked Codex agents.",
        "rating_method": "Each agent allocates exactly 1,000 nonnegative integer points jointly across the complete candidate set.",
        "blinding": "Each agent used only its separately shuffled neutral packet and prompt, without classifications, source data, other packets or ratings, analysis code, or figures.",
        "important_limitation": "The passes use separate agents from the same model family; this is not cross-model validation.",
        "n_raters": 3,
        "points_per_task_context_rater": 1000,
        "structural_choice_set": "all 22 vignettes, with the two DG-C wording revisions",
        "within_game_choice_sets": {"DG": 6, "UG": 8, "TG": 8},
        "raw_rows": len(combined), "averaged_rows": len(means),
    }
    (HERE / "relative_rating_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Validated three blinded passes; wrote {len(combined)} raw and {len(means)} averaged rows")


if __name__ == "__main__":
    main()
