#!/usr/bin/env python3
"""Write exploratory Codex 1,000-point relative-similarity allocations.

These are new fixed-budget judgments, not a renormalization of the earlier
absolute 0--100 ratings.  The structural task allocates across all 30
vignettes.  The within-game task allocates across the six DG vignettes or the
twelve vignettes belonging to the target receiver game.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "codex_relative_allocations.csv"

FAMILY_IDS = {
    "DG": ["V07", "V28", "V04", "V02", "V22", "V17"],
    "UG": ["V20", "V13", "V15", "V14", "V27", "V09", "V12", "V05", "V08", "V10", "V23", "V21"],
    "TG": ["V18", "V06", "V29", "V25", "V03", "V11", "V30", "V24", "V01", "V26", "V16", "V19"],
}

# Within-game allocations.  Every context-specific list sums to 1,000 and is
# ordered as in FAMILY_IDS.  Each list was authored as a fixed-budget relative
# judgment, with all vignettes in the relevant choice set considered jointly.
WITHIN_ALLOCATIONS = {
    # DG: M-P, M-K, S-P, S-K, C-P, C-K
    "T05": [180, 160, 170, 155, 175, 160],  # Control
    "T11": [100, 220, 100, 230, 110, 240],  # Market
    "T12": [210, 120, 140, 110, 290, 130],  # Bonus
    "T07": [260, 100, 220, 100, 220, 100],  # Aid
    # UG: M-C(P,K), M-D(P,K), S-C(P,K), S-D(P,K), C-C(P,K), C-D(P,K)
    "T04": [90, 80, 75, 70, 90, 80, 80, 75, 95, 85, 90, 90],
    "T01": [45, 105, 40, 100, 40, 130, 35, 135, 50, 120, 55, 145],
    "T06": [120, 50, 90, 40, 100, 55, 95, 50, 150, 70, 120, 60],
    "T03": [150, 45, 100, 35, 130, 50, 120, 50, 115, 55, 100, 50],
    # TG: M-C(P,K), M-D(P,K), S-C(P,K), S-D(P,K), C-C(P,K), C-D(P,K)
    "T08": [90, 85, 75, 70, 85, 80, 80, 75, 100, 90, 90, 80],
    "T09": [50, 115, 45, 105, 45, 125, 40, 120, 55, 140, 45, 115],
    "T02": [115, 55, 90, 45, 100, 50, 90, 45, 170, 75, 110, 55],
    "T10": [145, 55, 105, 40, 125, 50, 110, 45, 115, 55, 105, 50],
}

# Direct structural-family budgets for the all-30 task.  The three entries are
# DG, UG, and TG and sum to 1,000.  The family budget is distributed over its
# individual vignettes using the same frame's relative profile.  This records a
# two-level fixed-budget judgment while preserving a valid all-30 allocation.
STRUCTURAL_FAMILY_BUDGETS = {
    "T05": {"DG": 520, "UG": 280, "TG": 200},
    "T11": {"DG": 480, "UG": 300, "TG": 220},
    "T12": {"DG": 500, "UG": 290, "TG": 210},
    "T07": {"DG": 500, "UG": 280, "TG": 220},
    "T04": {"DG": 160, "UG": 540, "TG": 300},
    "T01": {"DG": 160, "UG": 500, "TG": 340},
    "T06": {"DG": 180, "UG": 520, "TG": 300},
    "T03": {"DG": 180, "UG": 520, "TG": 300},
    "T08": {"DG": 130, "UG": 270, "TG": 600},
    "T09": {"DG": 120, "UG": 300, "TG": 580},
    "T02": {"DG": 130, "UG": 300, "TG": 570},
    "T10": {"DG": 130, "UG": 300, "TG": 570},
}

CONTEXT_BY_GAME_FRAME = {
    ("DG", "Control"): "T05",
    ("DG", "Market"): "T11",
    ("DG", "Bonus"): "T12",
    ("DG", "Aid"): "T07",
    ("UG", "Control"): "T04",
    ("UG", "Market"): "T01",
    ("UG", "Bonus"): "T06",
    ("UG", "Aid"): "T03",
    ("TG", "Control"): "T08",
    ("TG", "Market"): "T09",
    ("TG", "Bonus"): "T02",
    ("TG", "Aid"): "T10",
}


def largest_remainder(profile: list[int], total: int) -> list[int]:
    weights = np.asarray(profile, dtype=float)
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(int)
    remainder = int(total - base.sum())
    if remainder:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:remainder]] += 1
    result = base.tolist()
    if sum(result) != total:
        raise AssertionError("Largest-remainder allocation failed")
    return result


def profile_for_family(frame: str, family: str) -> list[int]:
    return WITHIN_ALLOCATIONS[CONTEXT_BY_GAME_FRAME[(family, frame)]]


def main() -> None:
    mapping = json.loads((HERE / "anonymization_map.json").read_text(encoding="utf-8"))
    contexts = mapping["contexts"]
    rows: list[dict[str, object]] = []

    for context_id, budgets in STRUCTURAL_FAMILY_BUDGETS.items():
        frame = contexts[context_id]["frame"]
        for family in ["DG", "UG", "TG"]:
            points = largest_remainder(profile_for_family(frame, family), budgets[family])
            for vignette_id, value in zip(FAMILY_IDS[family], points):
                rows.append(
                    {
                        "rater": "codex-relative-exploratory",
                        "replicate": 1,
                        "task": "structural_all30",
                        "context_id": context_id,
                        "vignette_id": vignette_id,
                        "points": value,
                    }
                )

    for context_id, points in WITHIN_ALLOCATIONS.items():
        family = contexts[context_id]["game"]
        for vignette_id, value in zip(FAMILY_IDS[family], points):
            rows.append(
                {
                    "rater": "codex-relative-exploratory",
                    "replicate": 1,
                    "task": "within_game",
                    "context_id": context_id,
                    "vignette_id": vignette_id,
                    "points": value,
                }
            )

    result = pd.DataFrame(rows)
    expected_rows = 12 * 30 + 4 * 6 + 8 * 12
    if len(result) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(result)}")
    sums = result.groupby(["task", "context_id"])["points"].sum()
    if not (sums == 1000).all():
        raise AssertionError(f"Every task-context must sum to 1,000:\n{sums[sums != 1000]}")
    if (result["points"] < 0).any():
        raise AssertionError("Points must be nonnegative")
    result.to_csv(OUTPUT, index=False)

    protocol = {
        "description": "New fixed-budget Codex relative-similarity judgments; not derived by normalizing the prior 0-100 ratings.",
        "important_limitation": "One exploratory Codex judgment. The primary agent was not blind to the private vignette classification, so external blinded reruns are required before paper use.",
        "replicates": 1,
        "points_per_task_context": 1000,
        "structural_choice_set": "all 30 vignettes",
        "within_game_choice_sets": {"DG": 6, "UG": 12, "TG": 12},
        "rows": len(result),
    }
    (HERE / "single_agent_relative_rating_protocol_legacy.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(result)} validated allocations to {OUTPUT}")


if __name__ == "__main__":
    main()
