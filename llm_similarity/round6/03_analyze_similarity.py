#!/usr/bin/env python3
"""Normalize round-6 similarity ratings and make the requested grouped plots."""

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
CATEGORY_ORDER = ["Moral", "Self-interest", "Cooperation"]
CATEGORY_SHORT = {"Moral": "M", "Self-interest": "S", "Cooperation": "C"}
SETTING_ORDER = ["P", "K"]
SETTING_LABEL = {"P": "Personal", "K": "Anonymous market"}
CATEGORY_COLOR = {"Moral": "#4C78A8", "Self-interest": "#E45756", "Cooperation": "#54A24B"}
SETTING_HATCH = {"P": "", "K": "//", "T": "xx"}


def load_inputs(ratings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path)
    mapping = json.loads((HERE / "anonymization_map.json").read_text(encoding="utf-8"))
    vignette_map = (
        pd.DataFrame.from_dict(mapping["vignettes"], orient="index")
        .rename_axis("vignette_id")
        .reset_index()
    )
    context_map = (
        pd.DataFrame.from_dict(mapping["contexts"], orient="index")
        .rename_axis("context_id")
        .reset_index()
    )
    required = {"provider", "model", "replicate", "context_id", "vignette_id", "rating"}
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise ValueError(f"Ratings file is missing columns: {missing}")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    if not ratings["rating"].between(0, 100).all():
        raise ValueError("Ratings must all lie in [0, 100]")
    keys = ["provider", "model", "replicate", "context_id", "vignette_id"]
    if ratings.duplicated(keys).any():
        raise ValueError("Duplicate provider-model-replicate-context-vignette rows found")
    unknown_contexts = sorted(set(ratings["context_id"]).difference(context_map["context_id"]))
    unknown_vignettes = sorted(set(ratings["vignette_id"]).difference(vignette_map["vignette_id"]))
    if unknown_contexts or unknown_vignettes:
        raise ValueError(
            "Ratings contain identifiers absent from the private map: "
            f"contexts={unknown_contexts}, vignettes={unknown_vignettes}"
        )
    return ratings, vignette_map, context_map


def validate_coverage(ratings: pd.DataFrame, context_map: pd.DataFrame) -> None:
    counts = ratings.groupby(["provider", "model", "replicate", "context_id"]).size()
    if not (counts == 30).all():
        raise ValueError(f"Every completed rating unit must contain 30 scores; bad counts:\n{counts[counts != 30]}")
    expected_contexts = set(context_map["context_id"])
    for unit, part in ratings.groupby(["provider", "model", "replicate"]):
        observed = set(part["context_id"])
        if observed != expected_contexts:
            raise ValueError(f"Incomplete context coverage for {unit}: missing={sorted(expected_contexts - observed)}")


def analysis_pool_mask(frame: pd.DataFrame) -> pd.Series:
    """DG uses DG vignettes; UG and TG use the combined UG+TG vignette pool."""
    dg_pool = (frame["game_context"] == "DG") & (frame["game_vignette"] == "DG")
    receiver_pool = (frame["game_context"].isin(["UG", "TG"])) & (
        frame["game_vignette"].isin(["UG", "TG"])
    )
    return dg_pool | receiver_pool


def compute_weights(
    ratings: pd.DataFrame, vignette_map: pd.DataFrame, context_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize each repetition before averaging, preventing a repetition that
    # uses a higher part of the 0--100 scale from receiving greater weight.
    raw = ratings.merge(context_map, on="context_id", validate="many_to_one").merge(
        vignette_map,
        on="vignette_id",
        validate="many_to_one",
        suffixes=("_context", "_vignette"),
    )
    unit = ["provider", "model", "replicate", "context_id"]
    raw["weight_all30"] = raw["rating"] / raw.groupby(unit)["rating"].transform("sum")
    raw["in_analysis_pool"] = analysis_pool_mask(raw)
    raw["weight_analysis_pool"] = np.nan
    pool = raw["in_analysis_pool"]
    pool_totals = raw.loc[pool].groupby(unit)["rating"].transform("sum")
    if (pool_totals <= 0).any():
        raise ValueError("At least one analysis-pool similarity total is zero")
    raw.loc[pool, "weight_analysis_pool"] = raw.loc[pool, "rating"] / pool_totals

    value_columns = ["rating", "weight_all30", "weight_analysis_pool"]
    by_model = (
        raw.groupby(
            ["provider", "model", "context_id", "vignette_id"],
            as_index=False,
            dropna=False,
        )[value_columns]
        .mean()
        .rename(columns={"rating": "model_mean_rating"})
    )
    means = (
        by_model.groupby(["context_id", "vignette_id"], as_index=False, dropna=False)[
            ["model_mean_rating", "weight_all30", "weight_analysis_pool"]
        ]
        .mean()
        .rename(columns={"model_mean_rating": "mean_rating"})
    )
    full = means.merge(context_map, on="context_id", validate="many_to_one").merge(
        vignette_map,
        on="vignette_id",
        validate="many_to_one",
        suffixes=("_context", "_vignette"),
    )
    full["in_analysis_pool"] = analysis_pool_mask(full)
    plot_values = full.loc[full["in_analysis_pool"]].copy()
    checks = plot_values.groupby("context_id")["weight_analysis_pool"].sum()
    if not np.allclose(checks.to_numpy(), 1.0):
        raise AssertionError(f"Analysis-pool weights do not sum to one:\n{checks}")
    return full, plot_values


def block_label(row: pd.Series) -> str:
    category = CATEGORY_SHORT[row["sender_category"]]
    if row["game_context"] == "DG":
        return category
    return f"{category}-{row['receiver_action']}"


def add_individual_plot_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["receiver_action"] = result["receiver_action"].fillna("None")
    result["block"] = result.apply(block_label, axis=1)
    result["bar_type"] = "vignette"
    result["bar_label"] = (
        result["game_vignette"].astype(str)
        + "-"
        + result["setting_code"].astype(str)
        + "\n"
        + result["vignette_id"].astype(str)
    )
    return result


def add_class_totals(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    individual = add_individual_plot_metadata(frame)
    group_columns = [
        column
        for column in [
            "context_id",
            "source_id_context",
            "game_context",
            "frame",
            "comparison",
            "block",
            "sender_category",
            "receiver_action",
        ]
        if column in individual.columns
    ]
    totals = individual.groupby(group_columns, as_index=False, dropna=False)[value_columns].sum()
    totals["bar_type"] = "total"
    totals["bar_label"] = "TOTAL"
    totals["vignette_id"] = "TOTAL"
    totals["source_id_vignette"] = "Class total"
    totals["game_vignette"] = "TOTAL"
    totals["setting_code"] = "T"
    totals["setting"] = "class total"
    totals["joint_action"] = "TOTAL"
    return pd.concat([individual, totals], ignore_index=True, sort=False)


def ordered_panel(frame: pd.DataFrame, game: str) -> pd.DataFrame:
    subset = frame.loc[frame["game_context"] == game].copy()
    subset["category_order"] = subset["sender_category"].map(
        {name: index for index, name in enumerate(CATEGORY_ORDER)}
    )
    subset["receiver_order"] = subset["receiver_action"].map({"None": 0, "C": 0, "D": 1}).fillna(0)
    subset["total_order"] = subset["bar_type"].map({"vignette": 0, "total": 1})
    subset["source_game_order"] = subset["game_vignette"].map({"DG": 0, "UG": 0, "TG": 1, "TOTAL": 2})
    subset["setting_order"] = subset["setting_code"].map({"P": 0, "K": 1, "T": 2})
    return subset.sort_values(
        ["category_order", "receiver_order", "total_order", "source_game_order", "setting_order"]
    )


def bar_positions(panel: pd.DataFrame) -> tuple[np.ndarray, list[float], list[str]]:
    positions: list[float] = []
    centers: list[float] = []
    labels: list[str] = []
    cursor = 0.0
    for block, group in panel.groupby("block", sort=False):
        block_positions = [cursor + index for index in range(len(group))]
        positions.extend(block_positions)
        centers.append(float(np.mean(block_positions)))
        labels.append(block)
        cursor += len(group) + 1.0
    return np.asarray(positions), centers, labels


def draw_panel(ax, panel: pd.DataFrame, value_column: str, game: str, difference: bool) -> None:
    positions, centers, labels = bar_positions(panel)
    values = panel[value_column].to_numpy() * 100
    colors = [CATEGORY_COLOR[category] for category in panel["sender_category"]]
    bars = ax.bar(positions, values, color=colors, edgecolor="#333333", linewidth=0.55, width=0.78)
    for bar, (_, row) in zip(bars, panel.iterrows()):
        bar.set_hatch(SETTING_HATCH[row["setting_code"]])
        if row["bar_type"] == "total":
            bar.set_linewidth(1.3)
    if difference:
        ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(centers, labels)
    pool_note = "6 DG vignettes" if game == "DG" else "24 UG+TG vignettes"
    ax.set_title(f"{game} ({pool_note})", loc="left", fontweight="bold")
    ax.set_ylabel("Percentage points" if difference else "Normalized similarity (%)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for position, value, (_, row) in zip(positions, values, panel.iterrows()):
        ax.text(
            position,
            value,
            row["bar_label"],
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.2,
            fontweight="bold" if row["bar_type"] == "total" else "normal",
        )


def make_figure(data: pd.DataFrame, kind: str, value_column: str, output_path: Path) -> None:
    titles = {
        "control": "Control: normalized similarity distribution",
        "market_control": "Market minus Control: change in normalized similarity",
        "aid_bonus": "Aid minus Bonus: change in normalized similarity",
    }
    difference = kind != "control"
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    panel_maxima: list[float] = []
    for ax, game in zip(axes, GAME_ORDER):
        panel = ordered_panel(data, game)
        draw_panel(ax, panel, value_column, game, difference)
        panel_maxima.append(float(panel[value_column].abs().max() * 100))
    if difference:
        limit = max(panel_maxima) * 1.28 if max(panel_maxima) else 1
        for ax in axes:
            ax.set_ylim(-limit, limit)
    else:
        limit = max(panel_maxima) * 1.22 if max(panel_maxima) else 1
        for ax in axes:
            ax.set_ylim(0, limit)
    axes[-1].set_xlabel("Representation class (individual vignettes followed by the class TOTAL)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9", edgecolor="#333333", hatch=SETTING_HATCH["P"]),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9", edgecolor="#333333", hatch=SETTING_HATCH["K"]),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9", edgecolor="#333333", hatch=SETTING_HATCH["T"], linewidth=1.3),
    ]
    legend = fig.legend(
        handles,
        [SETTING_LABEL["P"], SETTING_LABEL["K"], "Class total"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
    )
    title = fig.suptitle(titles[kind], fontsize=16, y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(legend, title))
    plt.close(fig)


def build_comparison_rows(plot_values: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for label, first, second in [
        ("Market-Control", "Market", "Control"),
        ("Aid-Bonus", "Aid", "Bonus"),
    ]:
        index_columns = [
            "game_context",
            "vignette_id",
            "source_id_vignette",
            "game_vignette",
            "setting",
            "setting_code",
            "sender_category",
            "receiver_action",
            "joint_action",
        ]
        wide = plot_values.pivot(
            index=index_columns,
            columns="frame",
            values=["weight_analysis_pool", "weight_all30", "mean_rating"],
        ).reset_index()
        wide.columns = [
            column if isinstance(column, str) else "_".join(str(part) for part in column if part)
            for column in wide.columns
        ]
        wide["comparison"] = label
        wide["difference_analysis_pool"] = (
            wide[f"weight_analysis_pool_{first}"] - wide[f"weight_analysis_pool_{second}"]
        )
        wide["difference_all30"] = wide[f"weight_all30_{first}"] - wide[f"weight_all30_{second}"]
        wide["raw_rating_difference"] = wide[f"mean_rating_{first}"] - wide[f"mean_rating_{second}"]
        value_columns = [
            f"weight_analysis_pool_{first}",
            f"weight_analysis_pool_{second}",
            f"weight_all30_{first}",
            f"weight_all30_{second}",
            f"mean_rating_{first}",
            f"mean_rating_{second}",
            "difference_analysis_pool",
            "difference_all30",
            "raw_rating_difference",
        ]
        comparisons.append(add_class_totals(wide, value_columns))
    result = pd.concat(comparisons, ignore_index=True)
    individual_checks = (
        result.loc[result["bar_type"] == "vignette"]
        .groupby(["game_context", "comparison"])["difference_analysis_pool"]
        .sum()
    )
    total_checks = (
        result.loc[result["bar_type"] == "total"]
        .groupby(["game_context", "comparison"])["difference_analysis_pool"]
        .sum()
    )
    if not np.allclose(individual_checks.to_numpy(), 0.0) or not np.allclose(total_checks.to_numpy(), 0.0):
        raise AssertionError("Comparison differences do not sum to zero")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=HERE / "similarity_ratings.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--skip-coverage-check", action="store_true", help="Only for synthetic testing")
    args = parser.parse_args()

    ratings, vignette_map, context_map = load_inputs(args.ratings)
    if not args.skip_coverage_check:
        validate_coverage(ratings, context_map)
    full, plot_values = compute_weights(ratings, vignette_map, context_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.output_dir / "similarity_normalized_all_vignettes.csv", index=False)

    family_shares = (
        full.groupby(
            ["context_id", "source_id_context", "game_context", "frame", "game_vignette"],
            as_index=False,
        )["weight_all30"]
        .sum()
        .rename(columns={"game_vignette": "vignette_family", "weight_all30": "family_weight_all30"})
    )
    family_shares.to_csv(args.output_dir / "structural_family_shares.csv", index=False)

    control_individual = plot_values.loc[plot_values["frame"] == "Control"].copy()
    control_rows = add_class_totals(
        control_individual,
        ["weight_analysis_pool", "weight_all30", "mean_rating"],
    )
    control_rows.to_csv(args.output_dir / "control_plot_values.csv", index=False)
    control_individual_sum = (
        control_rows.loc[control_rows["bar_type"] == "vignette"]
        .groupby("game_context")["weight_analysis_pool"]
        .sum()
    )
    control_total_sum = (
        control_rows.loc[control_rows["bar_type"] == "total"]
        .groupby("game_context")["weight_analysis_pool"]
        .sum()
    )
    if not np.allclose(control_individual_sum.to_numpy(), 1.0) or not np.allclose(
        control_total_sum.to_numpy(), 1.0
    ):
        raise AssertionError("Control individual bars or class totals do not sum to one")

    comparison_rows = build_comparison_rows(plot_values)
    comparison_rows.to_csv(args.output_dir / "difference_plot_values.csv", index=False)

    make_figure(
        control_rows,
        "control",
        "weight_analysis_pool",
        args.output_dir / "control_similarity_distribution.png",
    )
    make_figure(
        comparison_rows.loc[comparison_rows["comparison"] == "Market-Control"],
        "market_control",
        "difference_analysis_pool",
        args.output_dir / "market_minus_control_similarity_differences.png",
    )
    make_figure(
        comparison_rows.loc[comparison_rows["comparison"] == "Aid-Bonus"],
        "aid_bonus",
        "difference_analysis_pool",
        args.output_dir / "aid_minus_bonus_similarity_differences.png",
    )
    print(f"Wrote normalized data and three figures to {args.output_dir}")


if __name__ == "__main__":
    main()
