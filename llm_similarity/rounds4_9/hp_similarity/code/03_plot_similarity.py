"""Plot HP similarity means with participant-clustered 95% CIs.

The bar heights are response-level means of the three-rater similarity averages.
Confidence intervals use an intercept-only cluster-robust variance estimator,
clustering the four hypothetical-allocation responses by participant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hp_responses_classified_and_rated.csv"
OUTPUT = ROOT / "output"

CONDITION = {0: "Control HPs", 1: "Market HPs"}
REFERENCE = {
    "similarity_dg_kw_control_mean": "Control instructions",
    "similarity_dg_kw_market_mean": "Market instructions",
}
CATEGORY = {
    1: ("M", "Moral"),
    3: ("S", "Self-interest"),
    2: ("C", "Mutual Benefit / Cooperation"),
}

REFERENCE_COLOR = {
    "Control instructions": "#7F7F7F",
    "Market instructions": "#55A868",
}
CATEGORY_COLOR = {"M": "#4C78A8", "S": "#E45756", "C": "#54A24B"}


def validate(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "hp_response_id",
        "PROLIFIC_PID",
        "Market",
        "category_num",
        *REFERENCE,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(data) != 4800 or data["hp_response_id"].duplicated().any():
        raise ValueError("Expected 4,800 unique HP-response rows")
    data = data.copy()
    data["Market"] = pd.to_numeric(data["Market"], errors="raise").astype(int)
    data["category_num"] = pd.to_numeric(
        data["category_num"], errors="raise"
    ).astype(int)
    if not set(data["Market"]).issubset({0, 1}):
        raise ValueError("Market must be coded 0/1")
    for column in REFERENCE:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].between(0, 100).all():
            raise ValueError(f"{column} contains values outside 0--100")
    return data


def clustered_mean_ci(
    frame: pd.DataFrame, value_column: str
) -> dict[str, float | int]:
    """Return a row-weighted mean and CR1 SE clustered by participant."""
    subset = frame[["PROLIFIC_PID", value_column]].dropna()
    if subset.empty:
        raise ValueError(f"No observations for {value_column}")
    y = subset[value_column].to_numpy(dtype=float)
    mean = float(y.mean())
    residual = subset.assign(residual=y - mean)
    cluster_sums = residual.groupby("PROLIFIC_PID")["residual"].sum().to_numpy()
    n = len(subset)
    n_clusters = len(cluster_sums)
    if n_clusters < 2:
        raise ValueError("At least two participant clusters are required")
    variance = (n_clusters / (n_clusters - 1)) * np.square(cluster_sums).sum() / n**2
    se = float(np.sqrt(variance))
    critical_value = 1.96
    return {
        "mean": mean,
        "se_clustered": se,
        "ci_low": mean - critical_value * se,
        "ci_high": mean + critical_value * se,
        "n_responses": n,
        "n_participants": n_clusters,
    }


def source_condition_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_code, condition_label in CONDITION.items():
        condition_data = data.loc[data["Market"] == condition_code]
        for value_column, reference_label in REFERENCE.items():
            rows.append(
                {
                    "figure": "source_condition",
                    "source_condition_code": condition_code,
                    "source_condition": condition_label,
                    "reference": reference_label,
                    "category_num": pd.NA,
                    "category_short": pd.NA,
                    "category": pd.NA,
                    **clustered_mean_ci(condition_data, value_column),
                }
            )
    return pd.DataFrame(rows)


def category_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    condition_reference = {
        0: "similarity_dg_kw_control_mean",
        1: "similarity_dg_kw_market_mean",
    }
    for condition_code, condition_label in CONDITION.items():
        condition_data = data.loc[data["Market"] == condition_code]
        value_column = condition_reference[condition_code]
        for category_num, (category_short, category_label) in CATEGORY.items():
            category_data = condition_data.loc[
                condition_data["category_num"] == category_num
            ]
            rows.append(
                {
                    "figure": "category_within_condition",
                    "source_condition_code": condition_code,
                    "source_condition": condition_label,
                    "reference": REFERENCE[value_column],
                    "category_num": category_num,
                    "category_short": category_short,
                    "category": category_label,
                    **clustered_mean_ci(category_data, value_column),
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def error_sizes(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [frame["mean"] - frame["ci_low"], frame["ci_high"] - frame["mean"]]
    )


def plot_source_condition(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    x = np.arange(len(CONDITION), dtype=float)
    width = 0.34
    for reference_index, reference_label in enumerate(REFERENCE.values()):
        subset = (
            summary.loc[summary["reference"] == reference_label]
            .set_index("source_condition_code")
            .reindex(CONDITION)
            .reset_index()
        )
        positions = x + (reference_index - 0.5) * width
        bars = ax.bar(
            positions,
            subset["mean"],
            width,
            yerr=error_sizes(subset),
            capsize=4,
            label=reference_label,
            color=REFERENCE_COLOR[reference_label],
            edgecolor="#333333",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=9)
    upper = float(summary["ci_high"].max())
    ax.set_ylim(0, min(100, max(10, upper * 1.22)))
    ax.set_xticks(x, list(CONDITION.values()))
    ax.set_ylabel("Mean similarity rating (0--100)")
    ax.set_title("Similarity of hypothetical-allocation descriptions to DG-KW instructions")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_panels(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True)
    order = [1, 3, 2]
    upper = float(summary["ci_high"].max())
    y_limit = min(100, max(10, upper * 1.22))
    for ax, condition_code in zip(axes, CONDITION):
        subset = (
            summary.loc[summary["source_condition_code"] == condition_code]
            .set_index("category_num")
            .reindex(order)
            .reset_index()
        )
        short_labels = subset["category_short"].tolist()
        bars = ax.bar(
            short_labels,
            subset["mean"],
            yerr=error_sizes(subset),
            capsize=4,
            color=[CATEGORY_COLOR[label] for label in short_labels],
            edgecolor="#333333",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=9)
        ax.set_title(
            f"{CONDITION[condition_code]}: similarity to\n"
            f"{REFERENCE['similarity_dg_kw_control_mean' if condition_code == 0 else 'similarity_dg_kw_market_mean']}",
            fontweight="bold",
        )
        ax.set_xlabel("Classification")
        ax.set_ylim(0, y_limit)
        style_axis(ax)
    axes[0].set_ylabel("Mean similarity rating (0--100)")
    fig.suptitle("Instruction similarity by HP classification", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "M = Moral; S = Self-interest; C = Mutual Benefit / Cooperation",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = validate(pd.read_csv(DATA))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = source_condition_summary(data)
    category = category_summary(data)
    summary = pd.concat([source, category], ignore_index=True)
    summary.to_csv(OUTPUT / "hp_similarity_plot_values.csv", index=False)
    plot_source_condition(source, OUTPUT / "hp_similarity_by_source_condition.png")
    plot_category_panels(category, OUTPUT / "hp_similarity_by_category_within_condition.png")
    print(summary.to_string(index=False))
    print(f"Wrote HP similarity figures and values to {OUTPUT}")


if __name__ == "__main__":
    main()
