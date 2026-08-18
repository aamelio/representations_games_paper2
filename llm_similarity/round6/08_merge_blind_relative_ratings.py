#!/usr/bin/env python3
"""Validate and combine the three blinded 1,000-point rating passes."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUTS = {
    1: HERE / "relative_agent1_ratings.csv",
    2: HERE / "relative_agent2_ratings.csv",
    3: HERE / "relative_agent3_ratings.csv",
}
PACKETS = {
    rater: HERE / "relative_packets" / f"rater_{rater}_tasks.json"
    for rater in INPUTS
}
COMBINED_OUTPUT = HERE / "relative_ratings_3agents.csv"
MEAN_OUTPUT = HERE / "relative_ratings_3agent_means.csv"
AGREEMENT_OUTPUT = HERE / "relative_rater_agreement.csv"
PROTOCOL_OUTPUT = HERE / "relative_rating_protocol.json"
REQUIRED_COLUMNS = [
    "rater",
    "replicate",
    "task",
    "context_id",
    "vignette_id",
    "points",
]


def expected_pairs(packet_path: Path) -> set[tuple[str, str, str]]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    return {
        (task["stage"], task["context_id"], vignette["vignette_id"])
        for task in packet["tasks"]
        for vignette in task["vignettes"]
    }


def validate_one(path: Path, packet_path: Path, rater_number: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    if list(data.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {REQUIRED_COLUMNS}")
    if len(data) != 480:
        raise ValueError(f"{path.name}: expected 480 rows, found {len(data)}")
    expected_rater = f"relative-agent-{rater_number}"
    if set(data["rater"]) != {expected_rater}:
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
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(f"{path.name}: candidate mismatch; missing={missing}, extra={extra}")
    sums = data.groupby(["task", "context_id"])["points"].sum()
    if len(sums) != 24 or not (sums == 1000).all():
        raise ValueError(f"{path.name}: every one of the 24 tasks must sum to 1,000")
    return data


def main() -> None:
    frames = [
        validate_one(INPUTS[rater], PACKETS[rater], rater)
        for rater in sorted(INPUTS)
    ]
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 1440:
        raise AssertionError("Expected 1,440 combined rating rows")
    combined.to_csv(COMBINED_OUTPUT, index=False)

    means = (
        combined.groupby(["task", "context_id", "vignette_id"], as_index=False)
        .agg(
            mean_points=("points", "mean"),
            sd_points=("points", "std"),
            min_points=("points", "min"),
            max_points=("points", "max"),
            n_raters=("rater", "nunique"),
        )
    )
    if len(means) != 480 or not (means["n_raters"] == 3).all():
        raise AssertionError("Every candidate must have exactly three ratings")
    mean_sums = means.groupby(["task", "context_id"])["mean_points"].sum()
    if not np.allclose(mean_sums.to_numpy(), 1000.0):
        raise AssertionError("Averaged allocations must sum to 1,000")
    means.to_csv(MEAN_OUTPUT, index=False)

    agreement_rows: list[dict[str, object]] = []
    for (task, context_id), group in combined.groupby(["task", "context_id"]):
        wide = group.pivot(index="vignette_id", columns="rater", values="points")
        for rater_a, rater_b in combinations(sorted(wide.columns), 2):
            first = wide[rater_a]
            second = wide[rater_b]
            agreement_rows.append(
                {
                    "task": task,
                    "context_id": context_id,
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "pearson": first.corr(second),
                    "spearman": first.rank(method="average").corr(
                        second.rank(method="average")
                    ),
                    "mean_absolute_point_difference": (first - second).abs().mean(),
                    "total_variation_distance": 0.5
                    * (first - second).abs().sum()
                    / 1000,
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    if len(agreement) != 72:
        raise AssertionError("Expected three pairwise comparisons for each of 24 tasks")
    agreement.to_csv(AGREEMENT_OUTPUT, index=False)
    protocol = {
        "description": "Direct 1,000-point relative-similarity ratings averaged across three separately tasked Codex agents.",
        "rating_method": "Each agent allocates exactly 1,000 nonnegative integer points jointly across the complete candidate set.",
        "blinding": "Each agent was instructed to use only its separately shuffled neutral packet and prompt, and not to access classifications, other packets or ratings, earlier ratings, analysis code, or figures.",
        "important_limitation": "The passes were completed by separate agents from the same Codex model family; this is not cross-model validation.",
        "n_raters": 3,
        "points_per_task_context_rater": 1000,
        "structural_choice_set": "all 30 vignettes",
        "within_game_choice_sets": {"DG": 6, "UG": 12, "TG": 12},
        "raw_rows": len(combined),
        "averaged_rows": len(means),
    }
    PROTOCOL_OUTPUT.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Validated three blinded passes; wrote {len(combined)} rows to {COMBINED_OUTPUT}")
    print(f"Wrote {len(means)} averaged candidate ratings to {MEAN_OUTPUT}")
    print(f"Wrote {len(agreement)} within-task agreement diagnostics to {AGREEMENT_OUTPUT}")


if __name__ == "__main__":
    main()
