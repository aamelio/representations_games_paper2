#!/usr/bin/env python3
"""Additional six-representation histograms for the similarity exercise.

The script produces two parallel sets of descriptive figures.

1. LLM-implied representations. For each context and each of the six sender
   representations, average the absolute 0--100 ratings over eight vignettes and
   three independent conversations. Normalize the six means to sum to 100 within
   context. The eight contexts retained here are DG-KW, UG, and TG under Control
   and Market, plus the standalone Aid and Bonus stories.

2. Elicited representations. In UG and TG, interact the classified reason with a
   High/Low hypothetical-belief indicator. The median is calculated within game
   on the two conditions being compared (Control + Market or Bonus + Aid). High
   means strictly above the pooled median; observations equal to the median are
   assigned to Low so tied belief values stay together. DG-KW is reported
   separately using its three reason categories because no beliefs were elicited.

Inputs:
  pilot_recording.csv
  ../../replication_package/data/player1_all_categorized.xlsx

Outputs:
  additional_sim_histograms_output/*.png
  additional_sim_histograms_output/*.csv
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
P1_FILE = PAPER / "replication_package" / "data" / "player1_all_categorized.xlsx"
OUTPUT = HERE / "additional_sim_histograms_output"

CATEGORIES = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
REPRESENTATIONS = ["MH", "ML", "SH", "SL", "CH", "CL"]
REP_LABELS = {
    "MH": "Moral\nHigh belief",
    "ML": "Moral\nLow belief",
    "SH": "Self-interest\nHigh belief",
    "SL": "Self-interest\nLow belief",
    "CH": "Mutual Benefit/\nCooperation\nHigh belief",
    "CL": "Mutual Benefit/\nCooperation\nLow belief",
}
CATEGORY_TO_CODE = {
    "Moral": "M",
    "Self-interest": "S",
    "Mutual Benefit / Cooperation": "C",
}
CATEGORY_LABELS = {
    "Moral": "Moral",
    "Self-interest": "Self-interest",
    "Mutual Benefit / Cooperation": "Mutual Benefit/\nCooperation",
}

GAMES = ["dgkw", "ug", "tg"]
GAME_LABELS = {"dgkw": "DG-KW", "ug": "UG", "tg": "TG"}
GAME_COLORS = {"dgkw": "#4C566A", "ug": "#D08770", "tg": "#8FBCBB"}
STORY_LABELS = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}
STORY_COLORS = {0: "#4C566A", 1: "#D08770", 2: "#B48EAD", 4: "#A3BE8C"}

LLM_CONTEXTS = {
    "C-KW": ("Control", "dgkw"),
    "M-KW": ("Market", "dgkw"),
    "C-UG": ("Control", "ug"),
    "M-UG": ("Market", "ug"),
    "C-TG": ("Control", "tg"),
    "M-TG": ("Market", "tg"),
    "BONUS": ("Bonus", "story"),
    "AID": ("Aid", "story"),
}
COMPARISONS = [
    ("market_vs_control", 1, 0, "Market $-$ Control"),
    ("aid_vs_bonus", 4, 2, "Aid $-$ Bonus"),
]

FONT_SCALE = 1.45


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10 * FONT_SCALE,
            "axes.titlesize": 12 * FONT_SCALE,
            "axes.labelsize": 10 * FONT_SCALE,
            "xtick.labelsize": 8 * FONT_SCALE,
            "ytick.labelsize": 9 * FONT_SCALE,
            "legend.fontsize": 9 * FONT_SCALE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / filename, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, difference: bool = False) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if difference:
        ax.axhline(0, color="#444444", linewidth=1.0, zorder=2)


def grouped_bars(
    data: pd.DataFrame,
    series_order: list[str],
    series_labels: dict[str, str],
    series_colors: dict[str, str],
    value_col: str,
    title: str,
    ylabel: str,
    filename: str,
    difference: bool = False,
    category_order: list[str] | None = None,
    category_labels: dict[str, str] | None = None,
) -> None:
    order = category_order or REPRESENTATIONS
    labels = category_labels or REP_LABELS
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    x = np.arange(len(order))
    width = 0.78 / len(series_order)

    for idx, series in enumerate(series_order):
        subset = data[data["series"] == series].set_index("representation").reindex(order)
        positions = x + (idx - (len(series_order) - 1) / 2) * width
        ax.bar(
            positions,
            subset[value_col].to_numpy(),
            width=width * 0.94,
            color=series_colors[series],
            edgecolor="black",
            linewidth=0.6,
            label=series_labels[series],
            zorder=3,
        )

    style_axis(ax, difference=difference)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        frameon=False,
        ncol=len(series_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 1 + 0.13 * FONT_SCALE),
    )
    if not difference:
        ax.set_ylim(0, max(40, float(data[value_col].max()) * 1.18))
    fig.tight_layout()
    save_figure(fig, filename)


def build_llm_shares() -> pd.DataFrame:
    ratings = pd.read_csv(HERE / "pilot_recording.csv")
    ratings = ratings[ratings["context"].isin(LLM_CONTEXTS)].copy()
    ratings["representation"] = ratings["vignette_id"].str[:2]

    counts = ratings.groupby(["context", "representation"]).size()
    if not (counts == 24).all():
        raise ValueError("Each LLM context x representation must contain 8 vignettes x 3 sets.")

    shares = (
        ratings.groupby(["context", "representation"], as_index=False)["rating"]
        .mean()
        .rename(columns={"rating": "mean_absolute_score"})
    )
    totals = shares.groupby("context")["mean_absolute_score"].transform("sum")
    shares["share_pct"] = shares["mean_absolute_score"] / totals * 100
    shares[["condition", "game"]] = shares["context"].map(LLM_CONTEXTS).apply(pd.Series)
    shares["source"] = "LLM absolute ratings"

    sums = shares.groupby("context")["share_pct"].sum()
    if not np.allclose(sums, 100):
        raise AssertionError("Normalized LLM representation shares must sum to 100 by context.")
    return shares.sort_values(["context", "representation"])


def plot_llm(shares: pd.DataFrame) -> None:
    for condition in ["Control", "Market"]:
        plot = shares[shares["condition"] == condition].copy()
        plot["series"] = plot["game"]
        grouped_bars(
            plot,
            GAMES,
            GAME_LABELS,
            GAME_COLORS,
            "share_pct",
            f"LLM-implied representation distribution: {condition}",
            "Normalized similarity weight (%)",
            f"llm_{condition.lower()}_representation_distribution.png",
        )

    stories = shares[shares["condition"].isin(["Bonus", "Aid"])].copy()
    stories["series"] = stories["condition"]
    grouped_bars(
        stories,
        ["Bonus", "Aid"],
        {"Bonus": "Bonus", "Aid": "Aid"},
        {"Bonus": STORY_COLORS[2], "Aid": STORY_COLORS[4]},
        "share_pct",
        "LLM-implied representation distribution: Bonus and Aid",
        "Normalized similarity weight (%)",
        "llm_bonus_aid_representation_distribution.png",
    )

    wide = shares[shares["condition"].isin(["Control", "Market"])].pivot(
        index=["game", "representation"], columns="condition", values="share_pct"
    )
    market_diff = (wide["Market"] - wide["Control"]).rename("difference_pp").reset_index()
    market_diff["series"] = market_diff["game"]
    grouped_bars(
        market_diff,
        GAMES,
        GAME_LABELS,
        GAME_COLORS,
        "difference_pp",
        "Change in LLM-implied representations: Market minus Control",
        "Change in normalized weight (pp)",
        "llm_market_minus_control_representation_difference.png",
        difference=True,
    )

    story_wide = stories.pivot(
        index="representation", columns="condition", values="share_pct"
    )
    story_diff = (story_wide["Aid"] - story_wide["Bonus"]).rename("difference_pp").reset_index()
    story_diff["series"] = "difference"
    grouped_bars(
        story_diff,
        ["difference"],
        {"difference": "Aid $-$ Bonus"},
        {"difference": "#4C566A"},
        "difference_pp",
        "Change in LLM-implied representations: Aid minus Bonus",
        "Change in normalized weight (pp)",
        "llm_aid_minus_bonus_representation_difference.png",
        difference=True,
    )


def load_player1() -> pd.DataFrame:
    p1 = pd.read_excel(P1_FILE).copy()
    p1["game"] = p1["game"].astype(str).str.strip().str.lower()
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p1["beliefs_hp"] = pd.to_numeric(p1["beliefs_hp"], errors="coerce")
    p1["category"] = p1["category"].astype(str).str.strip()
    return p1[p1["category"].isin(CATEGORIES)].copy()


def build_elicited_strategic_shares(p1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison, treated, baseline, comparison_label in COMPARISONS:
        for game in ["ug", "tg"]:
            two = p1[(p1["game"] == game) & p1["story"].isin([treated, baseline])].copy()
            two = two.dropna(subset=["beliefs_hp"])
            pooled_median = float(two["beliefs_hp"].median())
            two["belief_level"] = np.where(
                two["beliefs_hp"] > pooled_median, "H", "L"
            )
            two["representation"] = (
                two["category"].map(CATEGORY_TO_CODE) + two["belief_level"]
            )

            for story in [baseline, treated]:
                condition = two[two["story"] == story]
                denominator = len(condition)
                counts = condition["representation"].value_counts()
                for representation in REPRESENTATIONS:
                    count = int(counts.get(representation, 0))
                    rows.append(
                        {
                            "comparison": comparison,
                            "comparison_label": comparison_label,
                            "game": game,
                            "story": story,
                            "condition": STORY_LABELS[story],
                            "representation": representation,
                            "count": count,
                            "n": denominator,
                            "share_pct": count / denominator * 100,
                            "pooled_median": pooled_median,
                            "tie_rule": "High if belief > median; Low if belief <= median",
                        }
                    )

    shares = pd.DataFrame(rows)
    sums = shares.groupby(["comparison", "game", "story"])["share_pct"].sum()
    if not np.allclose(sums, 100):
        raise AssertionError("Elicited strategic representation shares must sum to 100.")
    return shares


def plot_elicited_strategic(shares: pd.DataFrame) -> None:
    for story in [0, 1, 2, 4]:
        condition = STORY_LABELS[story]
        plot = shares[shares["story"] == story].copy()
        plot["series"] = plot["game"]
        grouped_bars(
            plot,
            ["ug", "tg"],
            GAME_LABELS,
            GAME_COLORS,
            "share_pct",
            f"Elicited representation distribution: {condition}",
            "Share of classified participants (%)",
            f"elicited_{condition.lower()}_representation_distribution.png",
        )

    for comparison, treated, baseline, comparison_label in COMPARISONS:
        two = shares[shares["comparison"] == comparison]
        wide = two.pivot(
            index=["game", "representation"], columns="story", values="share_pct"
        )
        diff = (wide[treated] - wide[baseline]).rename("difference_pp").reset_index()
        diff["series"] = diff["game"]
        grouped_bars(
            diff,
            ["ug", "tg"],
            GAME_LABELS,
            GAME_COLORS,
            "difference_pp",
            f"Change in elicited representations: {comparison_label}",
            "Change in participant share (pp)",
            f"elicited_{comparison}_representation_difference.png",
            difference=True,
        )


def build_elicited_dg_shares(p1: pd.DataFrame) -> pd.DataFrame:
    dg = p1[p1["game"] == "dgkw"].copy()
    rows = []
    for story in [0, 1, 2, 4]:
        condition = dg[dg["story"] == story]
        denominator = len(condition)
        counts = condition["category"].value_counts()
        for category in CATEGORIES:
            count = int(counts.get(category, 0))
            rows.append(
                {
                    "game": "dgkw",
                    "story": story,
                    "condition": STORY_LABELS[story],
                    "representation": category,
                    "count": count,
                    "n": denominator,
                    "share_pct": count / denominator * 100,
                }
            )
    shares = pd.DataFrame(rows)
    sums = shares.groupby("story")["share_pct"].sum()
    if not np.allclose(sums, 100):
        raise AssertionError("DG reason-category shares must sum to 100.")
    return shares


def plot_elicited_dg(shares: pd.DataFrame) -> None:
    for slug, story_order, title in [
        ("control_market", [0, 1], "DG-KW elicited reason distribution: Control and Market"),
        ("bonus_aid", [2, 4], "DG-KW elicited reason distribution: Bonus and Aid"),
    ]:
        plot = shares[shares["story"].isin(story_order)].copy()
        plot["series"] = plot["story"].astype(str)
        series_order = [str(story) for story in story_order]
        grouped_bars(
            plot,
            series_order,
            {str(story): STORY_LABELS[story] for story in story_order},
            {str(story): STORY_COLORS[story] for story in story_order},
            "share_pct",
            title,
            "Share of classified participants (%)",
            f"elicited_dg_{slug}_category_distribution.png",
            category_order=CATEGORIES,
            category_labels=CATEGORY_LABELS,
        )

    differences = []
    for comparison, treated, baseline, comparison_label in COMPARISONS:
        wide = shares[shares["story"].isin([treated, baseline])].pivot(
            index="representation", columns="story", values="share_pct"
        )
        diff = (wide[treated] - wide[baseline]).rename("difference_pp").reset_index()
        diff["series"] = comparison
        differences.append(diff)
    difference_data = pd.concat(differences, ignore_index=True)
    grouped_bars(
        difference_data,
        ["market_vs_control", "aid_vs_bonus"],
        {"market_vs_control": "Market $-$ Control", "aid_vs_bonus": "Aid $-$ Bonus"},
        {"market_vs_control": STORY_COLORS[1], "aid_vs_bonus": STORY_COLORS[4]},
        "difference_pp",
        "Change in DG-KW elicited reason distributions",
        "Change in participant share (pp)",
        "elicited_dg_category_distribution_differences.png",
        difference=True,
        category_order=CATEGORIES,
        category_labels=CATEGORY_LABELS,
    )


def main() -> None:
    set_plot_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    llm_shares = build_llm_shares()
    llm_shares.to_csv(OUTPUT / "llm_normalized_representation_shares.csv", index=False)
    plot_llm(llm_shares)

    p1 = load_player1()
    elicited_shares = build_elicited_strategic_shares(p1)
    elicited_shares.to_csv(OUTPUT / "elicited_representation_shares.csv", index=False)
    plot_elicited_strategic(elicited_shares)

    dg_shares = build_elicited_dg_shares(p1)
    dg_shares.to_csv(OUTPUT / "elicited_dg_category_shares.csv", index=False)
    plot_elicited_dg(dg_shares)

    print(f"Wrote figures and summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
