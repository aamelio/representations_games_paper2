from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent  # replication_package/
OUTPUT_DIR = ROOT / "data"
REPORT_DIR = ROOT
FIG_DIR = REPORT_DIR / "output" / "figures"
TABLE_DIR = REPORT_DIR / "output" / "tables"

P1_FILE = OUTPUT_DIR / "player1_all_categorized.xlsx"
P2_FILE = OUTPUT_DIR / "player2_all_categorized.xlsx"

COMPARISONS = [
    {
        "slug": "aid_vs_bonus",
        "title": "Aid vs Bonus",
        "treated": 4,
        "baseline": 2,
    },
    {
        "slug": "market_vs_control",
        "title": "Market vs Control",
        "treated": 1,
        "baseline": 0,
    },
]

GAME_LABELS = {
    "dgkw": "DG-KW",
    "ug": "UG",
    "tg": "TG",
}

GAME_COLORS = {
    "dgkw": "#4C566A",
    "ug": "#D08770",
    "tg": "#8FBCBB",
}

STORY_LABELS = {
    0: "Control",
    1: "Market",
    2: "Bonus",
    4: "Aid",
}

P1_CATEGORIES = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
P1_CATEGORY_LABELS = {
    "Moral": "Moral",
    "Self-interest": "Self-interest",
    "Mutual Benefit / Cooperation": "Mutual Benefit/\nCooperation",
}

P2_CATEGORIES = ["Moral good", "Moral bad", "Self-interest", "Mutual Benefit / Cooperation"]
P2_CATEGORY_LABELS = {
    "Moral good": "Moral good",
    "Moral bad": "Moral bad",
    "Self-interest": "Self-interest",
    "Mutual Benefit / Cooperation": "Mutual Benefit/\nCooperation",
}

P1_MODEL_ORDER = {
    "dgkw": ["Moral", "Mutual Benefit / Cooperation", "Self-interest"],
    "ug": ["Moral", "Mutual Benefit / Cooperation", "Self-interest"],
    "tg": ["Mutual Benefit / Cooperation", "Moral", "Self-interest"],
}

P2_MODEL_ORDER = {
    "ug": ["Mutual Benefit / Cooperation", "Self-interest", "Moral good", "Moral bad"],
    "tg": ["Moral good", "Mutual Benefit / Cooperation", "Self-interest", "Moral bad"],
}

PLAYER_GAMES = {
    "player1": ["dgkw", "ug", "tg"],
    "player2": ["ug", "tg"],
}


# Figures are drawn ~10in wide but included in the paper at ~5.3-6.0in, so every
# font is optically reduced by that factor on the page. Scaling the fonts here by
# the inverse lands them at roughly 8-9pt against the paper's 12pt body text.
FONT_SCALE = 1.55


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
            "xtick.labelsize": 9 * FONT_SCALE,
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


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, zero_line: bool = False) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0, color="#444444", linewidth=1.0, zorder=2)


def mean_ci_stats(values: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n_obs = len(series)

    if n_obs == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    mean = float(series.mean())
    sd = float(series.std(ddof=1)) if n_obs > 1 else 0.0
    se = sd / np.sqrt(n_obs) if n_obs > 0 else np.nan
    margin = 1.96 * se if pd.notna(se) else np.nan

    return {
        "mean": mean,
        "ci_low": mean - margin if pd.notna(margin) else np.nan,
        "ci_high": mean + margin if pd.notna(margin) else np.nan,
        "n": n_obs,
    }


def diff_in_means_stats(treated: pd.Series, baseline: pd.Series) -> dict[str, float]:
    treated_values = pd.to_numeric(treated, errors="coerce").dropna().astype(float)
    baseline_values = pd.to_numeric(baseline, errors="coerce").dropna().astype(float)

    if len(treated_values) == 0 or len(baseline_values) == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    estimate = float(treated_values.mean() - baseline_values.mean())
    treated_var = float(treated_values.var(ddof=1)) if len(treated_values) > 1 else 0.0
    baseline_var = float(baseline_values.var(ddof=1)) if len(baseline_values) > 1 else 0.0
    se = np.sqrt((treated_var / len(treated_values)) + (baseline_var / len(baseline_values)))
    margin = 1.96 * se

    return {
        "estimate": estimate,
        "ci_low": estimate - margin,
        "ci_high": estimate + margin,
    }


def symmetric_limit(values: list[float], minimum: float = 5.0, pad: float = 1.15) -> float:
    finite_values = [abs(v) for v in values if pd.notna(v)]
    if not finite_values:
        return minimum
    return max(minimum, max(finite_values) * pad)


def fosd_sign(
    baseline_shares: dict[str, float],
    treated_shares: dict[str, float],
    order_high_to_low: list[str],
) -> str:
    order_low_to_high = list(reversed(order_high_to_low))
    baseline_cum = 0.0
    treated_cum = 0.0
    treated_dominates = True
    baseline_dominates = True
    any_strict_treated = False
    any_strict_baseline = False

    for category in order_low_to_high:
        baseline_cum += baseline_shares.get(category, 0.0)
        treated_cum += treated_shares.get(category, 0.0)

        if treated_cum > baseline_cum + 1e-12:
            treated_dominates = False
        if treated_cum < baseline_cum - 1e-12:
            baseline_dominates = False
        if treated_cum < baseline_cum - 1e-12:
            any_strict_treated = True
        if treated_cum > baseline_cum + 1e-12:
            any_strict_baseline = True

    if treated_dominates and any_strict_treated:
        return "increase"
    if baseline_dominates and any_strict_baseline:
        return "decrease"
    return "ambiguous"


def load_player1() -> pd.DataFrame:
    df = pd.read_excel(P1_FILE).copy()
    df["game"] = df["game"].astype(str).str.strip().str.lower()
    df["story"] = pd.to_numeric(df["story"], errors="coerce")
    df["category"] = df["category"].astype(str).str.strip()
    df["share_sent"] = pd.to_numeric(df["share_sent"], errors="coerce")
    df["beliefs_hp"] = pd.to_numeric(df["beliefs_hp"], errors="coerce")
    df["sp_num"] = pd.to_numeric(df["sp_num"], errors="coerce")
    df = df[df["game"].isin(PLAYER_GAMES["player1"])].copy()
    df = df[df["category"].isin(P1_CATEGORIES)].copy()
    return df


def load_player2() -> pd.DataFrame:
    df = pd.read_excel(P2_FILE).copy()
    df["game"] = df["game"].astype(str).str.strip().str.lower()
    df["story"] = pd.to_numeric(df["story"], errors="coerce")
    df["category"] = df["category"].astype(str).str.strip()
    df["share_sent_hp"] = pd.to_numeric(df["share_sent_hp"], errors="coerce")
    df["choice_hp"] = pd.to_numeric(df["choice_hp"], errors="coerce")
    df["sp_num"] = pd.to_numeric(df["sp_num"], errors="coerce")
    df = df[df["game"].isin(PLAYER_GAMES["player2"])].copy()
    df = df[df["category"].isin(P2_CATEGORIES)].copy()
    return df


def load_player1_all_games() -> pd.DataFrame:
    df = pd.read_excel(P1_FILE).copy()
    df["story"] = pd.to_numeric(df["story"], errors="coerce")
    df["category"] = df["category"].astype(str).str.strip()
    df["sp_num"] = pd.to_numeric(df["sp_num"], errors="coerce")
    df = df[df["category"].isin(P1_CATEGORIES)].copy()
    return df


def load_player2_all_games() -> pd.DataFrame:
    df = pd.read_excel(P2_FILE).copy()
    df["story"] = pd.to_numeric(df["story"], errors="coerce")
    df["category"] = df["category"].astype(str).str.strip()
    df["sp_num"] = pd.to_numeric(df["sp_num"], errors="coerce")
    df = df[df["category"].isin(P2_CATEGORIES)].copy()
    return df


def build_control_category_distribution_figure(
    df: pd.DataFrame,
    player: str,
    categories: list[str],
    category_labels: dict[str, str],
    output_name: str,
) -> pd.DataFrame:
    rows = []

    for game in PLAYER_GAMES[player]:
        control = df[(df["story"] == 0) & (df["game"] == game)].copy()
        total = len(control)

        for category in categories:
            count = int((control["category"] == category).sum())
            share = count / total if total else np.nan
            se = np.sqrt(share * (1 - share) / total) if total else np.nan
            ci_low = share - 1.96 * se if pd.notna(se) else np.nan
            ci_high = share + 1.96 * se if pd.notna(se) else np.nan
            rows.append(
                {
                    "game": game,
                    "category": category,
                    "share_pct": share * 100 if pd.notna(share) else np.nan,
                    "ci_low_pct": ci_low * 100 if pd.notna(ci_low) else np.nan,
                    "ci_high_pct": ci_high * 100 if pd.notna(ci_high) else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2.2), 5.8))
    x = np.arange(len(categories))
    width = 0.72 / len(PLAYER_GAMES[player])

    for idx, game in enumerate(PLAYER_GAMES[player]):
        subset = summary[summary["game"] == game].set_index("category").reindex(categories)
        positions = x + (idx - (len(PLAYER_GAMES[player]) - 1) / 2) * width
        centers = subset["share_pct"].to_numpy()
        lower = centers - subset["ci_low_pct"].to_numpy()
        upper = subset["ci_high_pct"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.6,
            label=GAME_LABELS[game],
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )

    style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([category_labels[category] for category in categories])
    ax.set_ylabel("Share of responses (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{'Player 1' if player == 'player1' else 'Player 2'}: Control category distribution")
    ax.legend(frameon=False, ncol=len(PLAYER_GAMES[player]), loc="upper center", bbox_to_anchor=(0.5, 1 + 0.13 * FONT_SCALE))

    fig.tight_layout()
    save_figure(fig, output_name)
    return summary


def write_sp_category_table(
    df: pd.DataFrame,
    player_label: str,
    categories: list[str],
    category_labels: dict[str, str],
    output_name: str,
    caption: str,
    label: str,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    work["sp_group"] = np.where(
        work["sp_num"] > 1,
        "High social proximity",
        "Low social proximity",
    )

    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{{caption}}}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" + "c" * len(categories) + "}",
        "\\toprule",
        "Counterpart proximity & " + " & ".join(category_labels[category].replace("\n", "") for category in categories) + " \\\\",
        "\\midrule",
    ]

    for sp_group in ["Low social proximity", "High social proximity"]:
        row = [sp_group]
        for category in categories:
            subset = work[work["category"] == category]
            share = (subset["sp_group"] == sp_group).mean() if len(subset) else np.nan
            row.append(f"{share * 100:.1f}\\%")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{flushleft}",
            (
                f"\\footnotesize Notes: Pooled across all games and treatments for {player_label}. "
                "Columns report the distribution of counterpart proximity within each stated reason category and sum to 100\\%."
            ),
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )

    (TABLE_DIR / output_name).write_text("\n".join(lines), encoding="utf-8")


def build_player2_control_hp_outcome_figure(p2: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for game in PLAYER_GAMES["player2"]:
        control = p2[(p2["story"] == 0) & (p2["game"] == game)].copy()
        outcome_col = "choice_hp" if game == "ug" else "share_sent_hp"

        for category in P2_CATEGORIES:
            stats = mean_ci_stats(control.loc[control["category"] == category, outcome_col])
            rows.append(
                {
                    "game": game,
                    "category": category,
                    "mean_pct": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pct": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pct": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                }
            )

    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(P2_CATEGORIES))
    width = 0.72 / len(PLAYER_GAMES["player2"])

    for idx, game in enumerate(PLAYER_GAMES["player2"]):
        subset = summary[summary["game"] == game].set_index("category").reindex(P2_CATEGORIES)
        positions = x + (idx - (len(PLAYER_GAMES["player2"]) - 1) / 2) * width
        centers = subset["mean_pct"].to_numpy()
        lower = centers - subset["ci_low_pct"].to_numpy()
        upper = subset["ci_high_pct"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.6,
            label=GAME_LABELS[game],
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )

    style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([P2_CATEGORY_LABELS[category] for category in P2_CATEGORIES])
    ax.set_ylabel("Average hypothetical outcome (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Player 2: Control hypothetical outcome by category")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1 + 0.13 * FONT_SCALE))

    fig.tight_layout()
    save_figure(fig, "player2_control_hp_outcome_by_category_substantive_only.png")
    return summary


def build_player1_control_outcome_figure(p1: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for game in PLAYER_GAMES["player1"]:
        control = p1[(p1["story"] == 0) & (p1["game"] == game)].copy()

        for category in P1_CATEGORIES:
            stats = mean_ci_stats(control.loc[control["category"] == category, "share_sent"])
            rows.append(
                {
                    "game": game,
                    "category": category,
                    "mean_pct": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pct": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pct": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(P1_CATEGORIES))
    width = 0.72 / len(PLAYER_GAMES["player1"])

    for idx, game in enumerate(PLAYER_GAMES["player1"]):
        subset = summary[summary["game"] == game].set_index("category").reindex(P1_CATEGORIES)
        positions = x + (idx - (len(PLAYER_GAMES["player1"]) - 1) / 2) * width
        centers = subset["mean_pct"].to_numpy()
        lower = centers - subset["ci_low_pct"].to_numpy()
        upper = subset["ci_high_pct"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.6,
            label=GAME_LABELS[game],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )

    style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([P1_CATEGORY_LABELS[category] for category in P1_CATEGORIES])
    ax.set_ylabel("Average outcome (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Player 1: Control outcome by category")
    ax.legend(frameon=False, ncol=len(PLAYER_GAMES["player1"]), loc="upper center", bbox_to_anchor=(0.5, 1 + 0.13 * FONT_SCALE))

    fig.tight_layout()
    save_figure(fig, "player1_control_outcome_by_category_substantive_only.png")
    return summary


def build_player1_control_hp_beliefs_figure(p1: pd.DataFrame) -> pd.DataFrame:
    categories = ["Overall", *P1_CATEGORIES]
    display_labels = {
        "Overall": "Overall",
        **P1_CATEGORY_LABELS,
    }
    bar_colors = ["#9E9E9E", "#4C566A", "#D08770", "#8FBCBB"]

    rows = []
    for game in ["ug", "tg"]:
        control = p1[(p1["story"] == 0) & (p1["game"] == game)].copy()
        stats = mean_ci_stats(control["beliefs_hp"])
        rows.append(
            {
                "game": game,
                "category": "Overall",
                "mean_pct": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                "ci_low_pct": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                "ci_high_pct": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
            }
        )
        for category in P1_CATEGORIES:
            category_stats = mean_ci_stats(control.loc[control["category"] == category, "beliefs_hp"])
            rows.append(
                {
                    "game": game,
                    "category": category,
                    "mean_pct": category_stats["mean"] * 100 if pd.notna(category_stats["mean"]) else np.nan,
                    "ci_low_pct": category_stats["ci_low"] * 100 if pd.notna(category_stats["ci_low"]) else np.nan,
                    "ci_high_pct": category_stats["ci_high"] * 100 if pd.notna(category_stats["ci_high"]) else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)

    for ax, game in zip(axes, ["ug", "tg"]):
        subset = summary[summary["game"] == game].set_index("category").reindex(categories)
        x = np.arange(len(categories))
        centers = subset["mean_pct"].to_numpy()
        lower = centers - subset["ci_low_pct"].to_numpy()
        upper = subset["ci_high_pct"].to_numpy() - centers

        ax.bar(
            x,
            centers,
            width=0.68,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            x,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )
        style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([display_labels[category] for category in categories])
        ax.set_ylim(0, 100)
        ax.set_title(f"{GAME_LABELS[game]}: Player 1 hypothetical beliefs")

    axes[0].set_ylabel("Average hypothetical belief (%)")
    fig.tight_layout()
    save_figure(fig, "player1_control_hp_beliefs_by_game.png")
    return summary


def compute_representation_effects(
    df: pd.DataFrame,
    categories: list[str],
    games: list[str],
) -> pd.DataFrame:
    rows = []

    for comparison in COMPARISONS:
        for game in games:
            baseline = df[(df["story"] == comparison["baseline"]) & (df["game"] == game)].copy()
            treated = df[(df["story"] == comparison["treated"]) & (df["game"] == game)].copy()

            for category in categories:
                stats = diff_in_means_stats(
                    (treated["category"] == category).astype(float),
                    (baseline["category"] == category).astype(float),
                )
                rows.append(
                    {
                        "comparison_slug": comparison["slug"],
                        "game": game,
                        "category": category,
                        "estimate_pp": stats["estimate"] * 100 if pd.notna(stats["estimate"]) else np.nan,
                        "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                        "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def plot_representation_panel(
    ax: plt.Axes,
    effects: pd.DataFrame,
    categories: list[str],
    category_labels: dict[str, str],
    games: list[str],
    title: str,
    y_limit: float | None = None,
) -> None:
    x = np.arange(len(categories))
    width = 0.74 / len(games)

    panel_values = []
    for game in games:
        subset = effects[effects["game"] == game].set_index("category").reindex(categories)
        panel_values.extend(subset["ci_low_pp"].tolist())
        panel_values.extend(subset["ci_high_pp"].tolist())

    if y_limit is None:
        y_limit = symmetric_limit(panel_values, minimum=10.0, pad=1.12)

    for idx, game in enumerate(games):
        subset = effects[effects["game"] == game].set_index("category").reindex(categories)
        positions = x + (idx - (len(games) - 1) / 2) * width
        centers = subset["estimate_pp"].to_numpy()
        lower = centers - subset["ci_low_pp"].to_numpy()
        upper = subset["ci_high_pp"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.6,
            label=GAME_LABELS[game],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )

    style_axis(ax, zero_line=True)
    ax.set_xticks(x)
    ax.set_xticklabels([category_labels[category] for category in categories])
    ax.set_ylim(-y_limit, y_limit)
    ax.set_title(title)


def build_representation_figure(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    p1_effects = compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"])
    p2_effects = compute_representation_effects(p2, P2_CATEGORIES, PLAYER_GAMES["player2"])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    panel_specs = [
        (axes[0, 0], p1_effects, P1_CATEGORIES, P1_CATEGORY_LABELS, PLAYER_GAMES["player1"], "Aid vs Bonus: Player 1", "aid_vs_bonus"),
        (axes[0, 1], p2_effects, P2_CATEGORIES, P2_CATEGORY_LABELS, PLAYER_GAMES["player2"], "Aid vs Bonus: Player 2", "aid_vs_bonus"),
        (axes[1, 0], p1_effects, P1_CATEGORIES, P1_CATEGORY_LABELS, PLAYER_GAMES["player1"], "Market vs Control: Player 1", "market_vs_control"),
        (axes[1, 1], p2_effects, P2_CATEGORIES, P2_CATEGORY_LABELS, PLAYER_GAMES["player2"], "Market vs Control: Player 2", "market_vs_control"),
    ]

    for ax, frame, categories, labels, games, title, slug in panel_specs:
        plot_representation_panel(
            ax=ax,
            effects=frame[frame["comparison_slug"] == slug].copy(),
            categories=categories,
            category_labels=labels,
            games=games,
            title=title,
        )

    axes[0, 0].set_ylabel("Difference in category share (pp)")
    axes[1, 0].set_ylabel("Difference in category share (pp)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.015 * FONT_SCALE))
    fig.suptitle("Representation treatment effects", y=1.01, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "paper_representation_treatment_effects.png")


def compute_outcome_model_effects(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for comparison in COMPARISONS:
        for game in PLAYER_GAMES["player1"]:
            baseline = p1[(p1["story"] == comparison["baseline"]) & (p1["game"] == game)].copy()
            treated = p1[(p1["story"] == comparison["treated"]) & (p1["game"] == game)].copy()

            baseline_shares = baseline["category"].value_counts(normalize=True).to_dict()
            treated_shares = treated["category"].value_counts(normalize=True).to_dict()
            model_sign = fosd_sign(baseline_shares, treated_shares, P1_MODEL_ORDER[game])
            effect = diff_in_means_stats(treated["share_sent"], baseline["share_sent"])

            rows.append(
                {
                    "player": "player1",
                    "comparison_slug": comparison["slug"],
                    "game": game,
                    "estimate_pp": effect["estimate"] * 100 if pd.notna(effect["estimate"]) else np.nan,
                    "ci_low_pp": effect["ci_low"] * 100 if pd.notna(effect["ci_low"]) else np.nan,
                    "ci_high_pp": effect["ci_high"] * 100 if pd.notna(effect["ci_high"]) else np.nan,
                    "model_sign": model_sign,
                }
            )

        for game in PLAYER_GAMES["player2"]:
            baseline = p2[(p2["story"] == comparison["baseline"]) & (p2["game"] == game)].copy()
            treated = p2[(p2["story"] == comparison["treated"]) & (p2["game"] == game)].copy()

            baseline_shares = baseline["category"].value_counts(normalize=True).to_dict()
            treated_shares = treated["category"].value_counts(normalize=True).to_dict()
            model_sign = fosd_sign(baseline_shares, treated_shares, P2_MODEL_ORDER[game])

            outcome_col = "choice_hp" if game == "ug" else "share_sent_hp"
            effect = diff_in_means_stats(treated[outcome_col], baseline[outcome_col])

            rows.append(
                {
                    "player": "player2",
                    "comparison_slug": comparison["slug"],
                    "game": game,
                    "estimate_pp": effect["estimate"] * 100 if pd.notna(effect["estimate"]) else np.nan,
                    "ci_low_pp": effect["ci_low"] * 100 if pd.notna(effect["ci_low"]) else np.nan,
                    "ci_high_pp": effect["ci_high"] * 100 if pd.notna(effect["ci_high"]) else np.nan,
                    "model_sign": model_sign,
                }
            )

    return pd.DataFrame(rows)


def compute_player1_hp_belief_treatment_effects(p1: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison in COMPARISONS:
        for game in ["ug", "tg"]:
            baseline = p1[(p1["story"] == comparison["baseline"]) & (p1["game"] == game)].copy()
            treated = p1[(p1["story"] == comparison["treated"]) & (p1["game"] == game)].copy()
            stats = diff_in_means_stats(treated["beliefs_hp"], baseline["beliefs_hp"])
            rows.append(
                {
                    "comparison_slug": comparison["slug"],
                    "game": game,
                    "estimate_pp": stats["estimate"] * 100 if pd.notna(stats["estimate"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_player2_hp_level_summary(p2: pd.DataFrame, comparison: dict) -> pd.DataFrame:
    rows = []

    for story in [comparison["baseline"], comparison["treated"]]:
        for game in PLAYER_GAMES["player2"]:
            subset = p2[(p2["story"] == story) & (p2["game"] == game)].copy()
            outcome_col = "choice_hp" if game == "ug" else "share_sent_hp"
            stats = mean_ci_stats(subset[outcome_col])
            rows.append(
                {
                    "story": story,
                    "game": game,
                    "mean_pct": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pct": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pct": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def prediction_symbol(sign: str) -> str:
    if sign == "increase":
        return "+"
    if sign == "decrease":
        return "-"
    return ""


def is_significant(ci_low: float, ci_high: float) -> bool:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return False
    return ci_low > 0 or ci_high < 0


def plot_outcome_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    games: list[str],
    title: str,
    y_limit: float | None = None,
) -> None:
    subset = frame.set_index("game").reindex(games)
    x = np.arange(len(games))
    centers = subset["estimate_pp"].to_numpy()
    lower = centers - subset["ci_low_pp"].to_numpy()
    upper = subset["ci_high_pp"].to_numpy() - centers
    if y_limit is None:
        limit = symmetric_limit(
            subset["ci_low_pp"].tolist() + subset["ci_high_pp"].tolist(),
            minimum=6.0,
            pad=1.3,
        )
    else:
        limit = y_limit
    symbol_y = limit * 0.82

    ax.bar(
        x,
        centers,
        width=0.62,
        color=[GAME_COLORS[game] for game in games],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        x,
        centers,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
        zorder=4,
    )

    for idx, game in enumerate(games):
        sign = subset.loc[game, "model_sign"]
        symbol = prediction_symbol(sign) if is_significant(subset.loc[game, "ci_low_pp"], subset.loc[game, "ci_high_pp"]) else ""
        if symbol:
            ax.text(
                x[idx],
                symbol_y,
                symbol,
                ha="center",
                va="center",
                fontsize=15,
                color="#222222",
                fontweight="bold",
            )

    style_axis(ax, zero_line=True)
    ax.set_xticks(x)
    ax.set_xticklabels([GAME_LABELS[game] for game in games])
    ax.set_ylim(-limit, limit)
    ax.set_title(title)


def plot_simple_treatment_effect_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    games: list[str],
    title: str,
    y_limit: float | None = None,
) -> None:
    subset = frame.set_index("game").reindex(games)
    x = np.arange(len(games))
    centers = subset["estimate_pp"].to_numpy()
    lower = centers - subset["ci_low_pp"].to_numpy()
    upper = subset["ci_high_pp"].to_numpy() - centers
    if y_limit is None:
        limit = symmetric_limit(
            subset["ci_low_pp"].tolist() + subset["ci_high_pp"].tolist(),
            minimum=6.0,
            pad=1.3,
        )
    else:
        limit = y_limit

    ax.bar(
        x,
        centers,
        width=0.62,
        color=[GAME_COLORS[game] for game in games],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        x,
        centers,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
        zorder=4,
    )

    style_axis(ax, zero_line=True)
    ax.set_xticks(x)
    ax.set_xticklabels([GAME_LABELS[game] for game in games])
    ax.set_ylim(-limit, limit)
    ax.set_title(title)


def plot_player2_hp_level_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    comparison: dict,
    title: str,
) -> None:
    games = PLAYER_GAMES["player2"]
    stories = [comparison["baseline"], comparison["treated"]]
    x = np.arange(len(games))
    width = 0.34
    colors = ["#B0B0B0", "#D08770"]

    for idx, story in enumerate(stories):
        subset = summary[summary["story"] == story].set_index("game").reindex(games)
        positions = x + (idx - 0.5) * width
        centers = subset["mean_pct"].to_numpy()
        lower = centers - subset["ci_low_pct"].to_numpy()
        upper = subset["ci_high_pct"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6,
            label=STORY_LABELS[story],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )

    style_axis(ax, zero_line=False)
    ax.set_xticks(x)
    ax.set_xticklabels([GAME_LABELS[game] for game in games])
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, loc="upper right")


def build_player1_main_treatment_figure(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    p1_effects = compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"])
    outcome_effects = compute_outcome_model_effects(p1, p2)
    hp_effects = compute_player1_hp_belief_treatment_effects(p1)

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.4))

    for col_idx, comparison in enumerate(COMPARISONS):
        slug = comparison["slug"]
        title = comparison["title"]
        plot_representation_panel(
            axes[0, col_idx],
            p1_effects[p1_effects["comparison_slug"] == slug].copy(),
            P1_CATEGORIES,
            P1_CATEGORY_LABELS,
            PLAYER_GAMES["player1"],
            f"{title}: Representations",
        )
        plot_outcome_panel(
            axes[1, col_idx],
            outcome_effects[
                (outcome_effects["player"] == "player1")
                & (outcome_effects["comparison_slug"] == slug)
            ].copy(),
            PLAYER_GAMES["player1"],
            f"{title}: Outcome",
        )
        plot_simple_treatment_effect_panel(
            axes[2, col_idx],
            hp_effects[hp_effects["comparison_slug"] == slug].copy(),
            ["ug", "tg"],
            f"{title}: HP beliefs",
        )

    axes[0, 0].set_ylabel("Difference in category share (pp)")
    axes[1, 0].set_ylabel("Treatment effect on actual outcome (pp)")
    axes[2, 0].set_ylabel("Treatment effect on HP beliefs (pp)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.015 * FONT_SCALE))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "paper_section4_player1.png")


def compute_player1_comparison_joint_ylimits(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> dict[str, float]:
    """Per-panel y-limits shared by the two Player 1 comparison figures
    (paper_section4_player1_aid_vs_bonus / _market_vs_control). For each of the
    three panel types the limit is the autoscale rule used by the corresponding
    plot function (same minimum/pad), but applied to the union of the CI ranges
    across BOTH comparisons, so the two figures share identical per-panel scales
    and the larger Market effects are not visually understated."""
    p1_effects = compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"])
    outcome_effects = compute_outcome_model_effects(p1, p2)
    hp_effects = compute_player1_hp_belief_treatment_effects(p1)
    slugs = [comparison["slug"] for comparison in COMPARISONS]

    rep_vals: list[float] = []
    outcome_vals: list[float] = []
    hp_vals: list[float] = []
    for slug in slugs:
        rep = p1_effects[p1_effects["comparison_slug"] == slug]
        rep_vals.extend(rep["ci_low_pp"].tolist())
        rep_vals.extend(rep["ci_high_pp"].tolist())

        out = outcome_effects[
            (outcome_effects["player"] == "player1")
            & (outcome_effects["comparison_slug"] == slug)
        ]
        outcome_vals.extend(out["ci_low_pp"].tolist())
        outcome_vals.extend(out["ci_high_pp"].tolist())

        hp = hp_effects[hp_effects["comparison_slug"] == slug]
        hp_vals.extend(hp["ci_low_pp"].tolist())
        hp_vals.extend(hp["ci_high_pp"].tolist())

    return {
        "representations": symmetric_limit(rep_vals, minimum=10.0, pad=1.12),
        "outcome": symmetric_limit(outcome_vals, minimum=6.0, pad=1.3),
        "hp": symmetric_limit(hp_vals, minimum=6.0, pad=1.3),
    }


def build_player1_comparison_treatment_figure(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    comparison: dict,
    output_name: str,
    y_limits: dict[str, float] | None = None,
) -> None:
    p1_effects = compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"])
    outcome_effects = compute_outcome_model_effects(p1, p2)
    hp_effects = compute_player1_hp_belief_treatment_effects(p1)
    slug = comparison["slug"]
    title = comparison["title"]

    y_limits = y_limits or {}

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 11.0))

    plot_representation_panel(
        axes[0],
        p1_effects[p1_effects["comparison_slug"] == slug].copy(),
        P1_CATEGORIES,
        P1_CATEGORY_LABELS,
        PLAYER_GAMES["player1"],
        f"{title}: Representations",
        y_limit=y_limits.get("representations"),
    )
    plot_outcome_panel(
        axes[1],
        outcome_effects[
            (outcome_effects["player"] == "player1")
            & (outcome_effects["comparison_slug"] == slug)
        ].copy(),
        PLAYER_GAMES["player1"],
        f"{title}: Outcome",
        y_limit=y_limits.get("outcome"),
    )
    plot_simple_treatment_effect_panel(
        axes[2],
        hp_effects[hp_effects["comparison_slug"] == slug].copy(),
        ["ug", "tg"],
        f"{title}: HP beliefs",
        y_limit=y_limits.get("hp"),
    )

    axes[0].set_ylabel("Difference in category share (pp)")
    axes[1].set_ylabel("Treatment effect on actual outcome (pp)")
    axes[2].set_ylabel("Treatment effect on HP beliefs (pp)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.012 * FONT_SCALE))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, output_name)


def build_player2_appendix_treatment_figure(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    p2_effects = compute_representation_effects(p2, P2_CATEGORIES, PLAYER_GAMES["player2"])
    outcome_effects = compute_outcome_model_effects(p1, p2)
    hp_levels = {
        comparison["slug"]: compute_player2_hp_level_summary(p2, comparison)
        for comparison in COMPARISONS
    }

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.4))

    for col_idx, comparison in enumerate(COMPARISONS):
        slug = comparison["slug"]
        title = comparison["title"]
        plot_representation_panel(
            axes[0, col_idx],
            p2_effects[p2_effects["comparison_slug"] == slug].copy(),
            P2_CATEGORIES,
            P2_CATEGORY_LABELS,
            PLAYER_GAMES["player2"],
            f"{title}: Player 2 representations",
        )
        plot_outcome_panel(
            axes[1, col_idx],
            outcome_effects[
                (outcome_effects["player"] == "player2")
                & (outcome_effects["comparison_slug"] == slug)
            ].copy(),
            PLAYER_GAMES["player2"],
            f"{title}: Player 2 HP outcome",
        )
        plot_player2_hp_level_panel(
            axes[2, col_idx],
            hp_levels[slug],
            comparison,
            f"{title}: Player 2 hypothetical action",
        )

    axes[0, 0].set_ylabel("Difference in category share (pp)")
    axes[1, 0].set_ylabel("Treatment effect on HP outcome (pp)")
    axes[2, 0].set_ylabel("Average hypothetical outcome (%)")

    rep_handles, rep_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(rep_handles, rep_labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.012 * FONT_SCALE))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "appendix_player2_treatment_effects.png")


def build_outcome_model_figure(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    effects = compute_outcome_model_effects(p1, p2)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6))

    plot_outcome_panel(
        ax=axes[0, 0],
        frame=effects[(effects["player"] == "player1") & (effects["comparison_slug"] == "aid_vs_bonus")].copy(),
        games=PLAYER_GAMES["player1"],
        title="Player 1: Aid vs Bonus",
    )
    plot_outcome_panel(
        ax=axes[0, 1],
        frame=effects[(effects["player"] == "player1") & (effects["comparison_slug"] == "market_vs_control")].copy(),
        games=PLAYER_GAMES["player1"],
        title="Player 1: Market vs Control",
    )
    plot_outcome_panel(
        ax=axes[1, 0],
        frame=effects[(effects["player"] == "player2") & (effects["comparison_slug"] == "aid_vs_bonus")].copy(),
        games=PLAYER_GAMES["player2"],
        title="Player 2: Aid vs Bonus",
    )
    plot_outcome_panel(
        ax=axes[1, 1],
        frame=effects[(effects["player"] == "player2") & (effects["comparison_slug"] == "market_vs_control")].copy(),
        games=PLAYER_GAMES["player2"],
        title="Player 2: Market vs Control",
    )

    axes[0, 0].set_ylabel("Treatment effect on actual outcome (pp)")
    axes[1, 0].set_ylabel("Treatment effect on HP outcome (pp)")

    fig.suptitle("Outcome treatment effects with model-predicted direction", y=0.99, fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Bars show treatment effects. '+' and '-' mark clear model-predicted signs only when the corresponding outcome effect is statistically distinguishable from zero.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, "paper_outcome_treatment_effects_with_model.png")


def build_mixed_treatment_figure(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    comparison_slug: str,
    output_name: str,
    top_titles: tuple[str, str],
    bottom_titles: tuple[str, str],
    bottom_left_player: str,
    bottom_left_slug: str,
    bottom_right_player: str,
    bottom_right_slug: str,
) -> None:
    p1_effects = compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"])
    p2_effects = compute_representation_effects(p2, P2_CATEGORIES, PLAYER_GAMES["player2"])
    outcome_effects = compute_outcome_model_effects(p1, p2)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))

    plot_representation_panel(
        axes[0, 0],
        p1_effects[p1_effects["comparison_slug"] == comparison_slug].copy(),
        P1_CATEGORIES,
        P1_CATEGORY_LABELS,
        PLAYER_GAMES["player1"],
        top_titles[0],
    )
    plot_representation_panel(
        axes[0, 1],
        p2_effects[p2_effects["comparison_slug"] == comparison_slug].copy(),
        P2_CATEGORIES,
        P2_CATEGORY_LABELS,
        PLAYER_GAMES["player2"],
        top_titles[1],
    )

    plot_outcome_panel(
        axes[1, 0],
        outcome_effects[
            (outcome_effects["player"] == bottom_left_player)
            & (outcome_effects["comparison_slug"] == bottom_left_slug)
        ].copy(),
        PLAYER_GAMES[bottom_left_player],
        bottom_titles[0],
    )
    plot_outcome_panel(
        axes[1, 1],
        outcome_effects[
            (outcome_effects["player"] == bottom_right_player)
            & (outcome_effects["comparison_slug"] == bottom_right_slug)
        ].copy(),
        PLAYER_GAMES[bottom_right_player],
        bottom_titles[1],
    )

    axes[0, 0].set_ylabel("Difference in category share (pp)")
    left_ylabel = (
        "Treatment effect on actual outcome (pp)"
        if bottom_left_player == "player1"
        else "Treatment effect on HP outcome (pp)"
    )
    right_ylabel = (
        "Treatment effect on actual outcome (pp)"
        if bottom_right_player == "player1"
        else "Treatment effect on HP outcome (pp)"
    )

    axes[1, 0].set_ylabel(left_ylabel)
    if bottom_left_player != bottom_right_player:
        axes[1, 1].set_ylabel(right_ylabel)
        axes[1, 1].yaxis.set_label_position("right")
        axes[1, 1].yaxis.tick_right()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.015 * FONT_SCALE))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_name)


def write_figure_stats(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    p1_control_cats: pd.DataFrame,
    p2_control_cats: pd.DataFrame,
    p2_control_hp_outcomes: pd.DataFrame,
    p1_control_outcomes: pd.DataFrame,
    p1_control_hp_beliefs: pd.DataFrame,
) -> None:
    """Numeric log of every statistic rendered on this script's figures, so prose
    numbers citing them can be audited without reading pixels (added 2026-07-16).
    Control summaries are the exact DataFrames behind the figures; treatment-effect
    frames come from the same compute functions the figure builders call."""
    lines: list[str] = []

    def add(title: str, frame: pd.DataFrame) -> None:
        lines.append(f"=== {title} ===")
        lines.append(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines.append("")

    add("Control category shares, Player 1 (% of classified; fig player1_control_category_distribution)", p1_control_cats)
    add("Control category shares, Player 2 (% of classified; fig player2_control_category_distribution)", p2_control_cats)
    add("Control outcomes by category, Player 1 (%; fig player1_control_outcome_by_category)", p1_control_outcomes)
    add("Control hypothetical beliefs, Player 1 (%; fig player1_control_hp_beliefs_by_game)", p1_control_hp_beliefs)
    add("Control hypothetical outcomes by category, Player 2 (%; fig player2_control_hp_outcome_by_category)", p2_control_hp_outcomes)
    add("Treatment effects on category shares, Player 1 (pp; figs paper_section4_*)",
        compute_representation_effects(p1, P1_CATEGORIES, PLAYER_GAMES["player1"]))
    add("Treatment effects on category shares, Player 2 (pp; fig appendix_player2_category_shifts)",
        compute_representation_effects(p2, P2_CATEGORIES, PLAYER_GAMES["player2"]))
    add("Treatment effects on outcomes, both players (pp; figs paper_section4_* and player2_hypothetical)",
        compute_outcome_model_effects(p1, p2))
    add("Treatment effects on Player 1 hypothetical beliefs (pp; figs paper_section4_*)",
        compute_player1_hp_belief_treatment_effects(p1))
    for comparison in COMPARISONS:
        add(f"Player 2 hypothetical action levels, {comparison['title']} (%; fig player2_hypothetical_treatment_effects)",
            compute_player2_hp_level_summary(p2, comparison))

    # Residual "No clear justification" shares in control, from the raw files
    # (the loaders drop the residual before any figure is built).
    residual_rows = []
    for player_label, path, games in [
        ("player1", P1_FILE, ["dgkw", "dglt", "ug", "tg"]),
        ("player2", P2_FILE, PLAYER_GAMES["player2"]),
    ]:
        raw = pd.read_excel(path)
        raw["game"] = raw["game"].astype(str).str.strip().str.lower()
        raw["story"] = pd.to_numeric(raw["story"], errors="coerce")
        raw["category"] = raw["category"].astype(str).str.strip()
        for game in games:
            cell = raw[(raw["story"] == 0) & (raw["game"] == game)]
            share = float((cell["category"] == "No clear justification").mean() * 100) if len(cell) else float("nan")
            residual_rows.append({"player": player_label, "game": game, "residual_pct": share, "n_control": len(cell)})
    add('Residual "No clear justification" share in control (%)', pd.DataFrame(residual_rows))

    # All-responses treatment effects --- the estimand quoted in the paper's text and
    # in the E4/E5 tables. The figures use the classified sample (the category panels
    # require it, and each figure uses one sample throughout); cell by cell the two
    # conventions differ by less than one percentage point.
    raw1 = pd.read_excel(P1_FILE)
    raw2 = pd.read_excel(P2_FILE)
    for raw in (raw1, raw2):
        raw["game"] = raw["game"].astype(str).str.strip().str.lower()
        raw["story"] = pd.to_numeric(raw["story"], errors="coerce")
    all_rows = []
    for comparison in COMPARISONS:
        for game in PLAYER_GAMES["player1"]:
            base = raw1[(raw1["story"] == comparison["baseline"]) & (raw1["game"] == game)]
            trea = raw1[(raw1["story"] == comparison["treated"]) & (raw1["game"] == game)]
            outcomes = [("action", "share_sent")] + ([] if game == "dgkw" else [("hp_belief", "beliefs_hp")])
            for outcome_label, col in outcomes:
                stats = diff_in_means_stats(trea[col], base[col])
                all_rows.append(
                    {
                        "player": "player1",
                        "comparison_slug": comparison["slug"],
                        "game": game,
                        "outcome": outcome_label,
                        "estimate_pp": stats["estimate"] * 100 if pd.notna(stats["estimate"]) else np.nan,
                        "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                        "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    }
                )
        for game in PLAYER_GAMES["player2"]:
            base = raw2[(raw2["story"] == comparison["baseline"]) & (raw2["game"] == game)]
            trea = raw2[(raw2["story"] == comparison["treated"]) & (raw2["game"] == game)]
            col = "choice_hp" if game == "ug" else "share_sent_hp"
            stats = diff_in_means_stats(trea[col], base[col])
            all_rows.append(
                {
                    "player": "player2",
                    "comparison_slug": comparison["slug"],
                    "game": game,
                    "outcome": "hp_outcome",
                    "estimate_pp": stats["estimate"] * 100 if pd.notna(stats["estimate"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                }
            )
    add("Treatment effects on ALL responses (pp; the estimand quoted in the text and the E4/E5 tables)",
        pd.DataFrame(all_rows))

    # Social proximity (reasons measure, Player 1, classified sample): the control
    # association between the evoked counterpart's proximity and the action, and the
    # market-vs-control anonymity shift. High SP = sp_num > 1 (anonymous peer /
    # teammate / friend), the cut behind the SP tables.
    sp_rows = []
    for game in PLAYER_GAMES["player1"]:
        for arm_label, story in [("control", 0), ("market", 1)]:
            arm = p1[(p1["story"] == story) & (p1["game"] == game)]
            valid = arm[arm["sp_num"].notna()]
            high = valid[valid["sp_num"] > 1]
            low = valid[valid["sp_num"] <= 1]
            sp_rows.append(
                {
                    "game": game,
                    "condition": arm_label,
                    "high_sp_share_pct": len(high) / len(valid) * 100 if len(valid) else np.nan,
                    "no_mention_share_pct": (valid["sp_num"] == 0).mean() * 100 if len(valid) else np.nan,
                    "action_high_sp_pct": high["share_sent"].mean() * 100,
                    "action_low_sp_pct": low["share_sent"].mean() * 100,
                    "gap_pp": (high["share_sent"].mean() - low["share_sent"].mean()) * 100,
                    "n": len(valid),
                }
            )
    add("Social proximity, reasons measure (P1, classified): SP shares and action by SP group, control vs market",
        pd.DataFrame(sp_rows))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "control_treatment_figure_stats.txt"
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


def main() -> None:
    set_plot_style()
    p1 = load_player1()
    p2 = load_player2()
    p1_all_games = load_player1_all_games()
    p2_all_games = load_player2_all_games()
    p1_control_cats = build_control_category_distribution_figure(
        p1,
        "player1",
        P1_CATEGORIES,
        P1_CATEGORY_LABELS,
        "player1_control_category_distribution_substantive_only.png",
    )
    p2_control_cats = build_control_category_distribution_figure(
        p2,
        "player2",
        P2_CATEGORIES,
        P2_CATEGORY_LABELS,
        "player2_control_category_distribution_substantive_only.png",
    )
    p2_control_hp_outcomes = build_player2_control_hp_outcome_figure(p2)
    p1_control_outcomes = build_player1_control_outcome_figure(p1)
    p1_control_hp_beliefs = build_player1_control_hp_beliefs_figure(p1)
    write_sp_category_table(
        p1_all_games,
        "Player 1",
        P1_CATEGORIES,
        P1_CATEGORY_LABELS,
        "player1_sp_by_category.tex",
        "Player 1: Counterpart Proximity within Stated Reason Category",
        "tab:player1_sp_by_category",
    )
    write_sp_category_table(
        p2_all_games,
        "Player 2",
        P2_CATEGORIES,
        P2_CATEGORY_LABELS,
        "player2_sp_by_category.tex",
        "Player 2: Counterpart Proximity within Stated Reason Category",
        "tab:player2_sp_by_category",
    )
    build_representation_figure(p1, p2)
    build_outcome_model_figure(p1, p2)
    build_player1_main_treatment_figure(p1, p2)
    player1_comparison_ylimits = compute_player1_comparison_joint_ylimits(p1, p2)
    print(
        "Shared per-panel y-limits for paper_section4_player1_* figures "
        f"(symmetric ±): {player1_comparison_ylimits}"
    )
    build_player1_comparison_treatment_figure(
        p1,
        p2,
        COMPARISONS[0],
        "paper_section4_player1_aid_vs_bonus.png",
        y_limits=player1_comparison_ylimits,
    )
    build_player1_comparison_treatment_figure(
        p1,
        p2,
        COMPARISONS[1],
        "paper_section4_player1_market_vs_control.png",
        y_limits=player1_comparison_ylimits,
    )
    build_player2_appendix_treatment_figure(p1, p2)
    build_mixed_treatment_figure(
        p1,
        p2,
        comparison_slug="aid_vs_bonus",
        output_name="paper_figure3_mixed.png",
        top_titles=("Aid vs Bonus: Player 1 representations", "Aid vs Bonus: Player 2 representations"),
        bottom_titles=("Player 1: Aid vs Bonus", "Player 2: Aid vs Bonus"),
        bottom_left_player="player1",
        bottom_left_slug="aid_vs_bonus",
        bottom_right_player="player2",
        bottom_right_slug="aid_vs_bonus",
    )
    build_mixed_treatment_figure(
        p1,
        p2,
        comparison_slug="market_vs_control",
        output_name="paper_figure4_mixed.png",
        top_titles=("Market vs Control: Player 1 representations", "Market vs Control: Player 2 representations"),
        bottom_titles=("Player 1: Market vs Control", "Player 2: Market vs Control"),
        bottom_left_player="player1",
        bottom_left_slug="market_vs_control",
        bottom_right_player="player2",
        bottom_right_slug="market_vs_control",
    )
    write_figure_stats(
        p1,
        p2,
        p1_control_cats,
        p2_control_cats,
        p2_control_hp_outcomes,
        p1_control_outcomes,
        p1_control_hp_beliefs,
    )


if __name__ == "__main__":
    main()
