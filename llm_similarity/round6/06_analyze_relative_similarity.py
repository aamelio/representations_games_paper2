#!/usr/bin/env python3
"""Analyze the two-stage 1,000-point relative-similarity exercise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
GAME_ORDER = ["DG", "UG", "TG"]
FRAME_ORDER = ["Control", "Market", "Bonus", "Aid"]
FAMILY_COLOR = {"DG": "#7F7F7F", "UG": "#4C78A8", "TG": "#F58518"}
CATEGORY_COLOR = {"M": "#4C78A8", "S": "#E45756", "C": "#54A24B"}


def load_data(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
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
        raise ValueError(f"Relative ratings are missing columns: {missing}")
    if ratings.duplicated(["rater", "replicate", "task", "context_id", "vignette_id"]).any():
        raise ValueError("Duplicate relative-allocation rows found")
    ratings["points"] = pd.to_numeric(ratings["points"], errors="raise")
    if (ratings["points"] < 0).any():
        raise ValueError("Relative-allocation points must be nonnegative")
    sums = ratings.groupby(["rater", "replicate", "task", "context_id"])["points"].sum()
    if not (sums == 1000).all():
        raise ValueError(f"Every relative allocation must sum to 1,000:\n{sums[sums != 1000]}")
    merged = ratings.merge(contexts, on="context_id", validate="many_to_one").merge(
        vignettes, on="vignette_id", validate="many_to_one"
    )
    return merged


def class_label(row: pd.Series) -> str:
    category = {"Moral": "M", "Self-interest": "S", "Cooperation": "C"}[
        row["sender_category"]
    ]
    if row["target_game"] == "DG":
        return category
    return f"{category}-{row['receiver_action']}"


def structural_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = data.loc[data["task"] == "structural_all30"].copy()
    counts = structural.groupby(["context_id", "vignette_game"])["vignette_id"].nunique()
    expected = {"DG": 6, "UG": 12, "TG": 12}
    for (_, family), count in counts.items():
        if count != expected[family]:
            raise AssertionError(f"Unexpected {family} vignette count: {count}")
    family = (
        structural.groupby(
            ["rater", "replicate", "context_id", "target_game", "frame", "vignette_game"],
            as_index=False,
        )
        .agg(total_points=("points", "sum"), n_vignettes=("vignette_id", "nunique"))
    )
    family["mean_points_per_vignette"] = family["total_points"] / family["n_vignettes"]
    family = (
        family.groupby(
            ["context_id", "target_game", "frame", "vignette_game"], as_index=False
        )[["total_points", "n_vignettes", "mean_points_per_vignette"]]
        .mean()
        .rename(columns={"vignette_game": "vignette_family"})
    )
    family["normalized_family_mean_share"] = family["mean_points_per_vignette"] / (
        family.groupby("context_id")["mean_points_per_vignette"].transform("sum")
    )
    share_checks = family.groupby("context_id")["normalized_family_mean_share"].sum()
    if not np.allclose(share_checks.to_numpy(), 1.0):
        raise AssertionError("Normalized structural family means do not sum to one")
    comparisons = []
    for label, first, second in [
        ("Market-Control", "Market", "Control"),
        ("Aid-Bonus", "Aid", "Bonus"),
    ]:
        wide = family.pivot(
            index=["target_game", "vignette_family", "n_vignettes"],
            columns="frame",
            values=["total_points", "mean_points_per_vignette"],
        ).reset_index()
        wide.columns = [
            column if isinstance(column, str) else "_".join(str(part) for part in column if part)
            for column in wide.columns
        ]
        wide["comparison"] = label
        wide["difference_total_points"] = wide[f"total_points_{first}"] - wide[f"total_points_{second}"]
        wide["difference_mean_points"] = (
            wide[f"mean_points_per_vignette_{first}"]
            - wide[f"mean_points_per_vignette_{second}"]
        )
        comparisons.append(wide)
    comparison = pd.concat(comparisons, ignore_index=True)
    weighted_checks = comparison.groupby(["target_game", "comparison"])["difference_total_points"].sum()
    if not np.allclose(weighted_checks.to_numpy(), 0.0):
        raise AssertionError("Structural family point differences do not sum to zero")
    return family, comparison


def within_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = data.loc[data["task"] == "within_game"].copy()
    if not (within["target_game"] == within["vignette_game"]).all():
        raise AssertionError("Within-game task contains a vignette from the wrong game")
    within["class"] = within.apply(class_label, axis=1)
    classes = (
        within.groupby(
            ["rater", "replicate", "context_id", "target_game", "frame", "class"],
            as_index=False,
        )["points"]
        .sum()
    )
    classes["share"] = classes["points"] / 1000
    classes = (
        classes.groupby(["context_id", "target_game", "frame", "class"], as_index=False)[
            ["points", "share"]
        ]
        .mean()
    )
    checks = classes.groupby(["target_game", "frame"])["share"].sum()
    if not np.allclose(checks.to_numpy(), 1.0):
        raise AssertionError("Within-game class shares do not sum to one")
    comparisons = []
    for label, first, second in [
        ("Market-Control", "Market", "Control"),
        ("Aid-Bonus", "Aid", "Bonus"),
    ]:
        wide = classes.pivot(
            index=["target_game", "class"], columns="frame", values=["points", "share"]
        ).reset_index()
        wide.columns = [
            column if isinstance(column, str) else "_".join(str(part) for part in column if part)
            for column in wide.columns
        ]
        wide["comparison"] = label
        wide["difference_points"] = wide[f"points_{first}"] - wide[f"points_{second}"]
        wide["difference_share"] = wide[f"share_{first}"] - wide[f"share_{second}"]
        comparisons.append(wide)
    comparison = pd.concat(comparisons, ignore_index=True)
    difference_checks = comparison.groupby(["target_game", "comparison"])["difference_share"].sum()
    if not np.allclose(difference_checks.to_numpy(), 0.0):
        raise AssertionError("Within-game class-share differences do not sum to zero")
    return classes, comparison


def plot_structural_control(family: pd.DataFrame, path: Path) -> None:
    control = family.loc[family["frame"] == "Control"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(GAME_ORDER), dtype=float)
    width = 0.24
    for offset, vignette_family in enumerate(GAME_ORDER):
        values = []
        for target in GAME_ORDER:
            value = control.loc[
                (control["target_game"] == target)
                & (control["vignette_family"] == vignette_family),
                "normalized_family_mean_share",
            ].iloc[0]
            values.append(value * 100)
        positions = x + (offset - 1) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=f"{vignette_family} vignettes",
            color=FAMILY_COLOR[vignette_family],
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, GAME_ORDER)
    ax.set_xlabel("Target game in Control")
    ax.set_ylabel("Normalized similarity share (%)")
    ax.set_title("Structural similarity in Control (family-size adjusted)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def class_order(game: str) -> list[str]:
    return ["M", "S", "C"] if game == "DG" else ["M-C", "M-D", "S-C", "S-D", "C-C", "C-D"]


def plot_within_panels(data: pd.DataFrame, value_column: str, title: str, path: Path, difference: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    maxima = []
    for ax, game in zip(axes, GAME_ORDER):
        order = class_order(game)
        subset = data.loc[data["target_game"] == game].set_index("class").reindex(order).reset_index()
        values = subset[value_column].to_numpy() * 100
        colors = [CATEGORY_COLOR[label[0]] for label in order]
        bars = ax.bar(order, values, color=colors, edgecolor="#333333", linewidth=0.6)
        if difference:
            ax.axhline(0, color="#333333", linewidth=0.9)
        else:
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_title(game, fontweight="bold")
        ax.set_ylabel("Percentage points" if difference else "Allocated points (%)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        maxima.append(float(np.max(np.abs(values))))
    limit = max(maxima) * 1.25 if max(maxima) else 1
    for ax in axes:
        ax.set_ylim((-limit, limit) if difference else (0, limit))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings", type=Path, default=HERE / "relative_ratings_3agents.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "relative_output")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.ratings)
    family, structural_comparison = structural_outputs(data)
    classes, within_comparison = within_outputs(data)
    family.to_csv(args.output_dir / "structural_family_means.csv", index=False)
    structural_comparison.to_csv(
        args.output_dir / "structural_family_differences.csv", index=False
    )
    classes.to_csv(args.output_dir / "within_game_class_shares.csv", index=False)
    within_comparison.to_csv(
        args.output_dir / "within_game_class_differences.csv", index=False
    )

    plot_structural_control(
        family, args.output_dir / "relative_structural_control.png"
    )
    plot_within_panels(
        classes.loc[classes["frame"] == "Control"],
        "share",
        "Within-game relative similarity in Control",
        args.output_dir / "relative_within_game_control.png",
        difference=False,
    )
    plot_within_panels(
        within_comparison.loc[within_comparison["comparison"] == "Market-Control"],
        "difference_share",
        "Within-game relative similarity: Market minus Control",
        args.output_dir / "relative_within_game_market_minus_control.png",
        difference=True,
    )
    plot_within_panels(
        within_comparison.loc[within_comparison["comparison"] == "Aid-Bonus"],
        "difference_share",
        "Within-game relative similarity: Aid minus Bonus",
        args.output_dir / "relative_within_game_aid_minus_bonus.png",
        difference=True,
    )
    print(f"Wrote two-stage relative-similarity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
