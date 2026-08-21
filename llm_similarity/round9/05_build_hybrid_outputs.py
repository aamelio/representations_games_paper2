#!/usr/bin/env python3
"""Use round-9 ratings for DG and original round-8 ratings for UG/TG."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROUND8 = HERE.parent / "round8"
OUTPUT = HERE / "hybrid_output"


def load_analysis_module():
    path = HERE / "04_analyze_relative_similarity.py"
    spec = importlib.util.spec_from_file_location("round9_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_round(root: Path) -> pd.DataFrame:
    ratings = pd.read_csv(root / "relative_ratings_3agents.csv")
    mapping = json.loads((root / "anonymization_map.json").read_text(encoding="utf-8"))
    contexts = (
        pd.DataFrame.from_dict(mapping["contexts"], orient="index")
        .rename_axis("context_id")
        .reset_index()
        .rename(columns={"game": "target_game"})
    )
    vignettes = (
        pd.DataFrame.from_dict(mapping["vignettes"], orient="index")
        .rename_axis("vignette_id")
        .reset_index()
        .rename(columns={"game": "vignette_game"})
    )
    return ratings.merge(contexts, on="context_id", validate="many_to_one").merge(
        vignettes, on="vignette_id", validate="many_to_one"
    )


def combine_by_target_game(round8: pd.DataFrame, round9: pd.DataFrame) -> pd.DataFrame:
    old = round8.loc[round8["target_game"].isin(["UG", "TG"])].copy()
    old["ratings_source"] = "round8_original"
    new = round9.loc[round9["target_game"] == "DG"].copy()
    new["ratings_source"] = "round9_dg_c_neutral"
    return pd.concat([new, old], ignore_index=True)


def original_structural_point_estimates(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original normalize-after-averaging structural estimator."""
    structural = data.loc[data["task"] == "structural_all22"].copy()
    agent = structural.groupby(
        ["rater", "replicate", "context_id", "target_game", "frame", "vignette_game"],
        as_index=False,
    ).agg(total_points=("points", "sum"), n_vignettes=("vignette_id", "nunique"))
    agent["mean_points_per_vignette"] = agent["total_points"] / agent["n_vignettes"]
    pooled = agent.groupby(
        ["context_id", "target_game", "frame", "vignette_game"], as_index=False
    )["mean_points_per_vignette"].mean()
    pooled["original_point_estimate"] = pooled["mean_points_per_vignette"] / pooled.groupby(
        "context_id"
    )["mean_points_per_vignette"].transform("sum")
    return pooled.rename(columns={"vignette_game": "vignette_family"})[
        ["context_id", "target_game", "frame", "vignette_family", "original_point_estimate"]
    ]


def recenter_structural_intervals(summary: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Keep the original point estimate and use agent variation for CI width."""
    keys = ["context_id", "target_game", "frame", "vignette_family"]
    output = summary.merge(original_structural_point_estimates(data), on=keys, validate="one_to_one")
    half_width = (output["ci95_high"] - output["ci95_low"]) / 2
    output["normalized_family_mean_share"] = output.pop("original_point_estimate")
    output["ci95_low"] = output["normalized_family_mean_share"] - half_width
    output["ci95_high"] = output["normalized_family_mean_share"] + half_width
    return output


def main() -> None:
    analysis = load_analysis_module()
    round8_data = load_round(ROUND8)
    round9_data = load_round(HERE)

    r8_struct_agent, r8_struct, r8_struct_diff_agent, r8_struct_diff = analysis.structural_outputs(round8_data)
    r9_struct_agent, r9_struct, r9_struct_diff_agent, r9_struct_diff = analysis.structural_outputs(round9_data)
    r8_within_agent, r8_within, r8_within_diff_agent, r8_within_diff = analysis.within_outputs(round8_data)
    r9_within_agent, r9_within, r9_within_diff_agent, r9_within_diff = analysis.within_outputs(round9_data)

    r8_struct = recenter_structural_intervals(r8_struct, round8_data)
    r9_struct = recenter_structural_intervals(r9_struct, round9_data)

    structural_agent = combine_by_target_game(r8_struct_agent, r9_struct_agent)
    structural = combine_by_target_game(r8_struct, r9_struct)
    structural_diff_agent = combine_by_target_game(r8_struct_diff_agent, r9_struct_diff_agent)
    structural_diff = combine_by_target_game(r8_struct_diff, r9_struct_diff)
    within_agent = combine_by_target_game(r8_within_agent, r9_within_agent)
    within = combine_by_target_game(r8_within, r9_within)
    within_diff_agent = combine_by_target_game(r8_within_diff_agent, r9_within_diff_agent)
    within_diff = combine_by_target_game(r8_within_diff, r9_within_diff)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    structural_agent.to_csv(OUTPUT / "structural_family_agent_estimates.csv", index=False)
    structural.to_csv(OUTPUT / "structural_family_means.csv", index=False)
    structural_diff_agent.to_csv(OUTPUT / "structural_family_agent_differences.csv", index=False)
    structural_diff.to_csv(OUTPUT / "structural_family_differences.csv", index=False)
    within_agent.to_csv(OUTPUT / "within_game_class_agent_estimates.csv", index=False)
    within.to_csv(OUTPUT / "within_game_class_shares.csv", index=False)
    within_diff_agent.to_csv(OUTPUT / "within_game_class_agent_differences.csv", index=False)
    within_diff.to_csv(OUTPUT / "within_game_class_differences.csv", index=False)

    analysis.plot_structural_control(structural, OUTPUT / "relative_structural_control.png")
    analysis.plot_within_panels(
        within.loc[within["frame"] == "Control"],
        "share",
        "Within-game relative similarity in Control",
        OUTPUT / "relative_within_game_control.png",
        False,
    )
    analysis.plot_within_panels(
        within_diff.loc[within_diff["comparison"] == "Market-Control"],
        "difference_share",
        "Within-game relative similarity: Market minus Control",
        OUTPUT / "relative_within_game_market_minus_control.png",
        True,
    )
    analysis.plot_within_panels(
        within_diff.loc[within_diff["comparison"] == "Aid-Bonus"],
        "difference_share",
        "Within-game relative similarity: Aid minus Bonus",
        OUTPUT / "relative_within_game_aid_minus_bonus.png",
        True,
    )

    protocol = {
        "DG": "Round 9 ratings after neutralizing the two DG-C vignettes.",
        "UG": "Original round 8 ratings; no rerating used in the manuscript.",
        "TG": "Original round 8 ratings; no rerating used in the manuscript.",
        "confidence_intervals": (
            "Two-sided 95% Student-t intervals across the three agent-level estimates "
            "from the corresponding round (df=2)."
        ),
        "structural_point_estimator": (
            "Original estimator: average vignette points across agents, calculate family mean "
            "points per vignette, then normalize the three family means. CI widths use the "
            "variation in the three agent-specific normalized analogues."
        ),
        "interpretation": "Model-run variability, not participant-sampling uncertainty.",
    }
    (OUTPUT / "hybrid_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote DG-round9/UG-TG-round8 hybrid outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
