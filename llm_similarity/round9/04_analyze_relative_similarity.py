#!/usr/bin/env python3
"""Analyze round 9 and form 95% t-intervals across three model runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "relative_output"
GAME_ORDER = ["DG", "UG", "TG"]
FAMILY_COLOR = {"DG": "#7F7F7F", "UG": "#4C78A8", "TG": "#F58518"}
CATEGORY_COLOR = {"M": "#4C78A8", "S": "#E45756", "C": "#54A24B"}
CLASS_ORDER = {
    "DG": ["M", "S", "C"],
    "UG": ["M-C", "S-C", "C-C", "C-D"],
    "TG": ["M-C", "S-D", "C-C", "C-D"],
}
T_CRIT_95_DF2 = 4.302652729911275


def load_data() -> pd.DataFrame:
    ratings = pd.read_csv(HERE / "relative_ratings_3agents.csv")
    mapping = json.loads((HERE / "anonymization_map.json").read_text(encoding="utf-8"))
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
    required = {"rater", "replicate", "task", "context_id", "vignette_id", "points"}
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise ValueError(f"Missing rating columns: {missing}")
    if ratings.duplicated(["rater", "replicate", "task", "context_id", "vignette_id"]).any():
        raise ValueError("Duplicate rating rows")
    sums = ratings.groupby(["rater", "replicate", "task", "context_id"])["points"].sum()
    if not (sums == 1000).all():
        raise ValueError("Every relative allocation must sum to 1,000")
    return ratings.merge(contexts, on="context_id", validate="many_to_one").merge(
        vignettes, on="vignette_id", validate="many_to_one"
    )


def add_t_interval(
    data: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    estimate_name: str,
) -> pd.DataFrame:
    summary = data.groupby(group_columns, as_index=False).agg(
        **{
            estimate_name: (value_column, "mean"),
            "sd_across_agents": (value_column, "std"),
            "n_agents": (value_column, "count"),
        }
    )
    if not (summary["n_agents"] == 3).all():
        raise AssertionError("Every interval must use exactly three agent-level estimates")
    half_width = T_CRIT_95_DF2 * summary["sd_across_agents"] / np.sqrt(summary["n_agents"])
    summary["ci95_low"] = summary[estimate_name] - half_width
    summary["ci95_high"] = summary[estimate_name] + half_width
    return summary


def class_label(row: pd.Series) -> str:
    category = {"Moral": "M", "Self-interest": "S", "Cooperation": "C"}[
        row["sender_category"]
    ]
    return category if row["target_game"] == "DG" else f"{category}-{row['receiver_action']}"


def structural_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structural = data.loc[data["task"] == "structural_all22"].copy()
    expected = {"DG": 6, "UG": 8, "TG": 8}
    counts = structural.groupby(["context_id", "vignette_game"])["vignette_id"].nunique()
    for (_, family), count in counts.items():
        if count != expected[family]:
            raise AssertionError(f"Unexpected {family} vignette count: {count}")

    agent = structural.groupby(
        ["rater", "replicate", "context_id", "target_game", "frame", "vignette_game"],
        as_index=False,
    ).agg(total_points=("points", "sum"), n_vignettes=("vignette_id", "nunique"))
    agent["mean_points_per_vignette"] = agent["total_points"] / agent["n_vignettes"]
    agent["normalized_family_mean_share"] = agent["mean_points_per_vignette"] / agent.groupby(
        ["rater", "replicate", "context_id"]
    )["mean_points_per_vignette"].transform("sum")
    agent = agent.rename(columns={"vignette_game": "vignette_family"})
    if not np.allclose(
        agent.groupby(["rater", "replicate", "context_id"])["normalized_family_mean_share"].sum(),
        1,
    ):
        raise AssertionError("Agent-level normalized structural shares do not sum to one")

    summary = add_t_interval(
        agent,
        ["context_id", "target_game", "frame", "vignette_family"],
        "normalized_family_mean_share",
        "normalized_family_mean_share",
    )
    auxiliary = agent.groupby(
        ["context_id", "target_game", "frame", "vignette_family"], as_index=False
    ).agg(
        total_points=("total_points", "mean"),
        n_vignettes=("n_vignettes", "mean"),
        mean_points_per_vignette=("mean_points_per_vignette", "mean"),
    )
    summary = summary.merge(
        auxiliary,
        on=["context_id", "target_game", "frame", "vignette_family"],
        validate="one_to_one",
    )
    if not np.allclose(summary.groupby("context_id")["normalized_family_mean_share"].sum(), 1):
        raise AssertionError("Mean structural shares do not sum to one")

    wide = agent.pivot(
        index=["rater", "replicate", "target_game", "vignette_family"],
        columns="frame",
        values="normalized_family_mean_share",
    ).reset_index()
    agent_differences = []
    for comparison, first, second in [
        ("Market-Control", "Market", "Control"),
        ("Aid-Bonus", "Aid", "Bonus"),
    ]:
        part = wide[["rater", "replicate", "target_game", "vignette_family"]].copy()
        part["comparison"] = comparison
        part["difference_share"] = wide[first] - wide[second]
        agent_differences.append(part)
    agent_differences = pd.concat(agent_differences, ignore_index=True)
    differences = add_t_interval(
        agent_differences,
        ["target_game", "vignette_family", "comparison"],
        "difference_share",
        "difference_share",
    )
    if not np.allclose(
        agent_differences.groupby(["rater", "replicate", "target_game", "comparison"])[
            "difference_share"
        ].sum(),
        0,
    ):
        raise AssertionError("Agent-level structural differences do not sum to zero")
    return agent, summary, agent_differences, differences


def within_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    within = data.loc[data["task"] == "within_game"].copy()
    if not (within["target_game"] == within["vignette_game"]).all():
        raise AssertionError("Within-game task includes an incorrect game")
    within["class"] = within.apply(class_label, axis=1)
    observed = {
        game: sorted(within.loc[within["target_game"] == game, "class"].unique())
        for game in GAME_ORDER
    }
    expected = {game: sorted(labels) for game, labels in CLASS_ORDER.items()}
    if observed != expected:
        raise AssertionError(f"Unexpected within-game classes: {observed}")

    agent = within.groupby(
        ["rater", "replicate", "context_id", "target_game", "frame", "class"], as_index=False
    )["points"].sum()
    agent["share"] = agent["points"] / 1000
    if not np.allclose(
        agent.groupby(["rater", "replicate", "target_game", "frame"])["share"].sum(), 1
    ):
        raise AssertionError("Agent-level within-game shares do not sum to one")
    summary = add_t_interval(
        agent,
        ["context_id", "target_game", "frame", "class"],
        "share",
        "share",
    )
    summary["points"] = summary["share"] * 1000
    if not np.allclose(summary.groupby(["target_game", "frame"])["share"].sum(), 1):
        raise AssertionError("Mean within-game shares do not sum to one")

    wide = agent.pivot(
        index=["rater", "replicate", "target_game", "class"], columns="frame", values="share"
    ).reset_index()
    agent_differences = []
    for comparison, first, second in [
        ("Market-Control", "Market", "Control"),
        ("Aid-Bonus", "Aid", "Bonus"),
    ]:
        part = wide[["rater", "replicate", "target_game", "class"]].copy()
        part["comparison"] = comparison
        part["difference_share"] = wide[first] - wide[second]
        agent_differences.append(part)
    agent_differences = pd.concat(agent_differences, ignore_index=True)
    differences = add_t_interval(
        agent_differences,
        ["target_game", "class", "comparison"],
        "difference_share",
        "difference_share",
    )
    differences["difference_points"] = differences["difference_share"] * 1000
    if not np.allclose(
        agent_differences.groupby(["rater", "replicate", "target_game", "comparison"])[
            "difference_share"
        ].sum(),
        0,
    ):
        raise AssertionError("Agent-level within-game differences do not sum to zero")
    return agent, summary, agent_differences, differences


def error_lengths(subset: pd.DataFrame, value_column: str) -> np.ndarray:
    values = subset[value_column].to_numpy(dtype=float)
    lower = values - subset["ci95_low"].to_numpy(dtype=float)
    upper = subset["ci95_high"].to_numpy(dtype=float) - values
    return np.vstack([lower, upper]) * 100


def plot_structural_control(summary: pd.DataFrame, path: Path) -> None:
    control = summary.loc[summary["frame"] == "Control"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(GAME_ORDER), dtype=float)
    width = 0.24
    interval_extrema = []
    for offset, family in enumerate(GAME_ORDER):
        subset = (
            control.loc[control["vignette_family"] == family]
            .set_index("target_game")
            .reindex(GAME_ORDER)
            .reset_index()
        )
        values = subset["normalized_family_mean_share"].to_numpy() * 100
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            yerr=error_lengths(subset, "normalized_family_mean_share"),
            capsize=4,
            label=f"{family} vignettes",
            color=FAMILY_COLOR[family],
            edgecolor="#333333",
            linewidth=0.6,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capthick": 1.0},
        )
        ax.bar_label(bars, fmt="%.1f", padding=8, fontsize=8)
        interval_extrema.extend((subset["ci95_low"] * 100).tolist())
        interval_extrema.extend((subset["ci95_high"] * 100).tolist())
    ax.set_xticks(x, GAME_ORDER)
    ax.set_xlabel("Target game in Control")
    ax.set_ylabel("Normalized similarity share (%)")
    ax.set_title("Structural similarity in Control (family-size adjusted)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(min(0, min(interval_extrema) * 1.08), max(interval_extrema) * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_within_panels(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    path: Path,
    difference: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    extrema = []
    for ax, game in zip(axes, GAME_ORDER):
        order = CLASS_ORDER[game]
        subset = data.loc[data["target_game"] == game].set_index("class").reindex(order).reset_index()
        if subset[[value_column, "ci95_low", "ci95_high"]].isna().any().any():
            raise AssertionError(f"Missing plotted value for {game}")
        values = subset[value_column].to_numpy() * 100
        bars = ax.bar(
            order,
            values,
            yerr=error_lengths(subset, value_column),
            capsize=4,
            color=[CATEGORY_COLOR[label[0]] for label in order],
            edgecolor="#333333",
            linewidth=0.6,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capthick": 1.0},
        )
        if difference:
            ax.axhline(0, color="#333333", linewidth=0.9)
        else:
            ax.bar_label(bars, fmt="%.1f", padding=8, fontsize=8)
        ax.set_title(game, fontweight="bold")
        ax.set_ylabel("Percentage points" if difference else "Allocated points (%)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        extrema.extend((subset["ci95_low"] * 100).tolist())
        extrema.extend((subset["ci95_high"] * 100).tolist())
    if difference:
        limit = max(abs(min(extrema)), abs(max(extrema))) * 1.12
        for ax in axes:
            ax.set_ylim(-limit, limit)
    else:
        lower = min(0, min(extrema) * 1.08)
        upper = max(extrema) * 1.15
        for ax in axes:
            ax.set_ylim(lower, upper)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    structural_agent, structural, structural_diff_agent, structural_diff = structural_outputs(data)
    within_agent, within, within_diff_agent, within_diff = within_outputs(data)

    structural_agent.to_csv(OUTPUT_DIR / "structural_family_agent_estimates.csv", index=False)
    structural.to_csv(OUTPUT_DIR / "structural_family_means.csv", index=False)
    structural_diff_agent.to_csv(OUTPUT_DIR / "structural_family_agent_differences.csv", index=False)
    structural_diff.to_csv(OUTPUT_DIR / "structural_family_differences.csv", index=False)
    within_agent.to_csv(OUTPUT_DIR / "within_game_class_agent_estimates.csv", index=False)
    within.to_csv(OUTPUT_DIR / "within_game_class_shares.csv", index=False)
    within_diff_agent.to_csv(OUTPUT_DIR / "within_game_class_agent_differences.csv", index=False)
    within_diff.to_csv(OUTPUT_DIR / "within_game_class_differences.csv", index=False)

    protocol = {
        "estimand": "Mean of the three separately blinded agent-level aggregate shares or paired frame differences.",
        "interval": "Two-sided 95% Student-t interval across agent-level estimates.",
        "n_agents": 3,
        "degrees_of_freedom": 2,
        "t_critical": T_CRIT_95_DF2,
        "interpretation": "Variability across model runs, not participant-sampling uncertainty.",
        "limitation": "With only three model runs, intervals are necessarily imprecise and sensitive to any single run.",
    }
    (OUTPUT_DIR / "model_run_ci_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    plot_structural_control(structural, OUTPUT_DIR / "relative_structural_control.png")
    plot_within_panels(
        within.loc[within["frame"] == "Control"],
        "share",
        "Within-game relative similarity in Control",
        OUTPUT_DIR / "relative_within_game_control.png",
        False,
    )
    plot_within_panels(
        within_diff.loc[within_diff["comparison"] == "Market-Control"],
        "difference_share",
        "Within-game relative similarity: Market minus Control",
        OUTPUT_DIR / "relative_within_game_market_minus_control.png",
        True,
    )
    plot_within_panels(
        within_diff.loc[within_diff["comparison"] == "Aid-Bonus"],
        "difference_share",
        "Within-game relative similarity: Aid minus Bonus",
        OUTPUT_DIR / "relative_within_game_aid_minus_bonus.png",
        True,
    )
    print(f"Wrote round-9 estimates, model-run intervals, and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
