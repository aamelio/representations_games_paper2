"""Build the analysis-ready within-subject HP panel.

The source studies contain two pre-choice HP elicitations per participant and
four hypothetical allocation anchors per elicitation. This script converts the
wide survey exports to one row per participant x elicitation x anchor and joins
the allocation and post-choice reason observed after that elicitation.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[2]
WITHIN_ROOT = PROJECT_ROOT / "within"
OUT = ROOT / "within_subject"
INPUT_DIR = OUT / "input"

CONTROL_DB = WITHIN_ROOT / "control" / "output" / "db.xlsx"
MARKET_DB = WITHIN_ROOT / "market" / "output" / "db.xlsx"
OUTCOMES = WITHIN_ROOT / "output" / "within_all_long_categorized.xlsx"

DOLLAR = "$"
ANCHOR_COLUMNS = {
    ("kw", 0): {
        12.0: f"memory_kw_{DOLLAR}0",
        8.0: f"memory_kw_{DOLLAR}4",
        6.0: f"memory_kw_{DOLLAR}6",
        4.0: f"memory_kw_{DOLLAR}8",
    },
    ("lt", 0): {
        12.0: f"memory_lt_-{DOLLAR}4",
        8.0: f"memory_lt_{DOLLAR}0",
        6.0: f"memory_lt_{DOLLAR}2",
        4.0: f"memory_lt_{DOLLAR}4",
    },
    ("kw", 1): {
        12.0: f"memory_kw_{DOLLAR}0",
        8.0: f"memory_kw_-{DOLLAR}4",
        6.0: f"memory_kw_-{DOLLAR}6",
        4.0: f"memory_kw_-{DOLLAR}8",
    },
    ("lt", 1): {
        12.0: f"memory_lt_{DOLLAR}4",
        8.0: f"memory_lt_{DOLLAR}0",
        6.0: f"memory_lt_-{DOLLAR}2",
        4.0: f"memory_lt_-{DOLLAR}4",
    },
}
CONTROL_MARKET_COLUMNS = {
    ("kw", 0): {
        12.0: f"memory_kw_control_{DOLLAR}0",
        8.0: f"memory_kw_control_{DOLLAR}4",
        6.0: f"memory_kw_control_{DOLLAR}6",
        4.0: f"memory_kw_control_{DOLLAR}8",
    },
    ("lt", 0): {
        12.0: f"memory_lt_control_-{DOLLAR}4",
        8.0: f"memory_lt_control_{DOLLAR}0",
        6.0: f"memory_lt_control_{DOLLAR}2",
        4.0: f"memory_lt_control_{DOLLAR}4",
    },
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_key(value: object) -> str:
    return clean_text(value).lower()


def build_control_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for source_row, row in data.reset_index(drop=True).iterrows():
        pid = str(row["PROLIFIC_PID"]).strip()
        kw_first = int(row["kw_first"])
        for game in ["kw", "lt"]:
            trial_order = 1 if (game == "kw") == (kw_first == 1) else 2
            for anchor, column in ANCHOR_COLUMNS[(game, 0)].items():
                rows.append({
                    "source_study": "control_kwlt",
                    "source_row": source_row,
                    "PROLIFIC_PID": pid,
                    "subject_id": f"control_kwlt::{pid}",
                    "game": game,
                    "Market": 0,
                    "trial_order": trial_order,
                    "anchor_allocation_kept": anchor,
                    "memory_source_column": column,
                    "memory": clean_text(row[column]),
                })
    return rows


def build_market_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for source_row, row in data.reset_index(drop=True).iterrows():
        pid = str(row["PROLIFIC_PID"]).strip()
        game = str(row["treatment"]).strip().lower()
        market_first = int(row["market_first"])
        subject_id = f"market_control::{game}::{pid}"
        for market in [0, 1]:
            trial_order = 1 if (market == 1) == (market_first == 1) else 2
            columns = (
                CONTROL_MARKET_COLUMNS[(game, 0)]
                if market == 0
                else ANCHOR_COLUMNS[(game, 1)]
            )
            for anchor, column in columns.items():
                rows.append({
                    "source_study": "market_control",
                    "source_row": source_row,
                    "PROLIFIC_PID": pid,
                    "subject_id": subject_id,
                    "game": game,
                    "Market": market,
                    "trial_order": trial_order,
                    "anchor_allocation_kept": anchor,
                    "memory_source_column": column,
                    "memory": clean_text(row[column]),
                })
    return rows


def prepare_outcomes() -> pd.DataFrame:
    outcomes = pd.read_excel(OUTCOMES).copy()
    outcomes["PROLIFIC_PID"] = outcomes["PROLIFIC_PID"].astype(str).str.strip()
    outcomes["game"] = outcomes["game"].astype(str).str.lower().str.strip()
    outcomes["Market"] = pd.to_numeric(outcomes["story"], errors="raise").astype(int)
    outcomes["actual_allocation_kept"] = pd.to_numeric(
        outcomes["allocation"], errors="raise"
    )
    outcomes["actual_share_sent"] = 1.0 - outcomes["actual_allocation_kept"] / 12.0
    outcomes["actual_reason"] = outcomes["reasons"].map(clean_text)
    outcomes["actual_reason_category"] = outcomes["category"].astype(str).str.strip()
    outcomes["actual_reason_category_num"] = pd.to_numeric(
        outcomes["category_num"], errors="coerce"
    ).astype("Int64")
    keep = [
        "PROLIFIC_PID",
        "source_study",
        "game",
        "Market",
        "trial_order",
        "actual_allocation_kept",
        "actual_share_sent",
        "actual_reason",
        "actual_reason_category",
        "actual_reason_category_num",
    ]
    outcomes["source_study"] = outcomes["design"]
    return outcomes[keep]


def validate_panel(panel: pd.DataFrame) -> None:
    expected_rows = 2 * 4 * panel["subject_id"].nunique()
    if len(panel) != expected_rows:
        raise ValueError(f"Expected {expected_rows} panel rows; found {len(panel)}.")
    per_elicitation = panel.groupby(
        ["subject_id", "trial_order"], observed=True
    ).size()
    if not per_elicitation.eq(4).all():
        raise ValueError("Every participant-period must have four anchor rows.")
    anchors = panel.groupby(
        ["subject_id", "trial_order"], observed=True
    )["anchor_allocation_kept"].apply(lambda x: tuple(sorted(x)))
    if not anchors.map(lambda values: values == (4.0, 6.0, 8.0, 12.0)).all():
        raise ValueError("Unexpected hypothetical-allocation anchors.")
    if panel[
        [
            "actual_allocation_kept",
            "actual_share_sent",
            "actual_reason_category",
        ]
    ].isna().any().any():
        raise ValueError("Outcome merge produced missing allocation or reason fields.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    control = pd.read_excel(CONTROL_DB)
    market = pd.read_excel(MARKET_DB)
    rows = build_control_rows(control) + build_market_rows(market)
    panel = pd.DataFrame(rows)
    panel["kw_dummy"] = (panel["game"] == "kw").astype(int)
    panel["market_x_kw"] = panel["Market"] * panel["kw_dummy"]
    panel["treatment_cell"] = np.select(
        [
            (panel["game"] == "kw") & (panel["Market"] == 0),
            (panel["game"] == "lt") & (panel["Market"] == 0),
            (panel["game"] == "kw") & (panel["Market"] == 1),
            (panel["game"] == "lt") & (panel["Market"] == 1),
        ],
        ["kw_control", "lt_control", "kw_market", "lt_market"],
        default="",
    )
    panel["anchor_share_sent"] = (
        12.0 - panel["anchor_allocation_kept"]
    ) / 12.0
    panel["memory_key"] = panel["memory"].map(normalized_key)
    outcomes = prepare_outcomes()
    panel = panel.merge(
        outcomes,
        on=[
            "PROLIFIC_PID",
            "source_study",
            "game",
            "Market",
            "trial_order",
        ],
        how="left",
        validate="many_to_one",
    )
    panel = panel.sort_values(
        ["source_study", "subject_id", "trial_order", "anchor_allocation_kept"]
    ).reset_index(drop=True)
    panel.insert(0, "within_hp_id", [f"WHP{index:05d}" for index in range(1, len(panel) + 1)])
    validate_panel(panel)
    panel.to_csv(INPUT_DIR / "within_hp_panel_unclassified.csv", index=False)

    unique = (
        panel.loc[panel["memory_key"].ne(""), ["memory_key", "memory"]]
        .drop_duplicates("memory_key")
        .sort_values("memory_key")
        .reset_index(drop=True)
    )
    unique.insert(
        0,
        "classification_id",
        [f"WCLS{index:05d}" for index in range(1, len(unique) + 1)],
    )
    unique.to_csv(INPUT_DIR / "within_hp_unique_texts.csv", index=False)

    summary = pd.DataFrame([
        {"statistic": "participants", "value": panel["subject_id"].nunique()},
        {"statistic": "elicitation_periods", "value": panel[["subject_id", "trial_order"]].drop_duplicates().shape[0]},
        {"statistic": "potential_hp_texts", "value": len(panel)},
        {"statistic": "nonempty_hp_texts", "value": panel["memory_key"].ne("").sum()},
        {"statistic": "unique_nonempty_hp_texts", "value": len(unique)},
        {"statistic": "empty_hp_texts", "value": panel["memory_key"].eq("").sum()},
    ])
    summary.to_csv(INPUT_DIR / "within_hp_panel_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
