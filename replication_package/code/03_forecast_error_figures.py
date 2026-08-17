"""
05_beliefs_graphs.py

Creates two families of figures:

1. Sensitivity graphs for Player 1 beliefs vs. share sent.
2. Story-by-game two-panel figures that match the externally added PNGs:
   top panel: average Player 2 outcome and average beliefs by P1 share sent
   bottom panel: Player 1 share-sent distribution
3. A grouped forecast-error graph by treatment and game.

All figures are written to moral_cooperation_paymax/figures/.
"""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------
# User-facing settings
# ---------------------------------------------------------------------

P1_CATEGORIZED_INPUT = Path("data/player1_all_categorized.xlsx")
P1_INPUT = P1_CATEGORIZED_INPUT
P2_INPUT = Path("data/player2_all_categorized.xlsx")
OUTPUT_DIR = Path(".")
FIGURES_DIR = OUTPUT_DIR / "output" / "figures"
UG_CATEGORY_FORECAST_OUTPUT = Path("data/ug_forecast_error_by_category.csv")
TG_CATEGORY_FORECAST_CM_OUTPUT = Path("data/tg_forecast_error_by_category_control_market.csv")
TG_CATEGORY_FORECAST_OUTPUT = Path("data/tg_forecast_error_by_category.csv")

# Keep categories whose within-game share is at least 5%.
MIN_CATEGORY_SHARE = 0.05

# Use conventional OLS standard errors by default, matching Stata's plain reg.
# To use robust standard errors instead, set COV_TYPE = "HC1".
COV_TYPE: Optional[str] = None

# Graph settings.
# The two forecast-error figures used in the paper take their fonts from these
# rcParams. They were previously the matplotlib defaults, i.e. sans-serif and
# optically shrunk to ~6pt once included at 0.96\textwidth. FONT_SCALE plays the
# same role as in script 01: drawn ~10in wide, included at ~6.5in.
FONT_SCALE = 1.35

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 10 * FONT_SCALE,
        "xtick.labelsize": 9 * FONT_SCALE,
        "ytick.labelsize": 9 * FONT_SCALE,
        "legend.fontsize": 9 * FONT_SCALE,
    }
)

X_GRID = np.linspace(0, 1, 250)
FIGSIZE = (8.5, 5.4)
TWO_PANEL_FIGSIZE = (15.25, 8.75)
DPI = 300
SHARE_DECIMALS = 6
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 12345
HP_REFERENCE_SHARE = round(1 / 3, SHARE_DECIMALS)
STORY_LABELS = {
    0: "Control",
    1: "Market",
    2: "Bonus",
    4: "Aid",
}

STORY_SLUGS = {
    0: "control",
    1: "market",
    2: "bonus",
    4: "aid",
}

UG_CATEGORY_ORDER = [
    "Moral",
    "Mutual Benefit / Cooperation",
    "Self-interest",
    "No clear justification",
]

UG_CATEGORY_LABELS = {
    "Moral": "Moral",
    "Mutual Benefit / Cooperation": "Mutual Benefit/\nCooperation",
    "Self-interest": "Self-interest",
    "No clear justification": "No clear\njustification",
}

FORECAST_GAME_COLORS = {
    "ug": "#4C72B0",
    "tg": "#DD8452",
}

COMBINED_GAME_COLORS = {
    "ug": "#D08770",
    "tg": "#8FBCBB",
}

BELIEF_TREATMENT_COMPARISONS = [
    {
        "stories": [0, 1],
        "title": "Control vs Market",
        "slug": "control_vs_market",
    },
    {
        "stories": [4, 2],
        "title": "Aid vs Bonus",
        "slug": "aid_vs_bonus",
    },
]

TREATMENT_COMPARISON_SPECS = [
    {
        "baseline_story": 0,
        "treated_story": 1,
        "title": "Market vs Control",
        "slug": "market_vs_control",
    },
    {
        "baseline_story": 2,
        "treated_story": 4,
        "title": "Aid vs Bonus",
        "slug": "aid_vs_bonus",
    },
]

BELIEF_TREATMENT_COLORS = {
    0: "#4C72B0",  # Control
    1: "#55A868",  # Market
    2: "#C44E52",  # Bonus
    4: "#8172B3",  # Aid
}

COMBINED_BELIEF_FORECAST_COMPARISONS = [
    {
        "stories": [0],
        "slug": "control",
        "title": "Control",
    },
    {
        "stories": [0, 1],
        "slug": "market_vs_control",
        "title": "Market vs Control",
    },
    {
        "stories": [2, 4],
        "slug": "aid_vs_bonus",
        "title": "Aid vs Bonus",
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def script_dir() -> Path:
    """Return the replication-package root (parent of code/), the anchor for all relative paths."""
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


def pct_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format an axis value in [0, 1] as a percentage."""
    return f"{int(round(x * 100))}%"


def pp_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format a value in percentage points."""
    return f"{int(round(x))}"


def normalize_share_values(series: pd.Series) -> pd.Series:
    """Round share values so equivalent offer levels merge reliably."""
    return pd.to_numeric(series, errors="coerce").round(SHARE_DECIMALS)


def share_tick_grid(game: str) -> np.ndarray:
    """Return the x locations used in the hand-added figures."""
    if game == "tg":
        return np.array([i / 12 for i in range(13)], dtype=float)
    if game == "ug":
        return np.array([i / 24 for i in range(25)], dtype=float)
    raise ValueError(f"Unsupported game: {game}")


def mean_ci(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Return the mean and a normal-approximation CI for one series."""
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    n = len(clean)

    if n == 0:
        return pd.Series({"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0})

    mean = float(clean.mean())
    if n == 1:
        ci_low = mean
        ci_high = mean
    else:
        z = 1.959963984540054
        se = float(clean.std(ddof=1) / np.sqrt(n))
        ci_low = mean - z * se
        ci_high = mean + z * se

    return pd.Series(
        {
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
        }
    )


def grouped_mean_ci(data: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return grouped means and 95% CIs by x_col for y_col."""
    grouped = (
        data[[x_col, y_col]]
        .dropna()
        .groupby(x_col, sort=True)[y_col]
        .apply(mean_ci)
        .unstack()
        .reset_index()
        .rename(columns={x_col: "x"})
        .sort_values("x")
    )

    for col in ["x", "mean", "ci_low", "ci_high"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    grouped["ci_low"] = grouped["ci_low"].clip(0, 1)
    grouped["ci_high"] = grouped["ci_high"].clip(0, 1)
    return grouped


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected columns exist and have the expected names."""
    required = ["game", "story", "beliefs", "share_sent"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "category" not in df.columns:
        raise ValueError("Missing required column: category")

    return df


def prepare_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common fields used across the plotting functions."""
    df = df.copy()
    df["game"] = df["game"].astype(str).str.lower().str.strip()
    df["story"] = pd.to_numeric(df["story"], errors="coerce")
    return df


def fill_tg_zero_beliefs_ctr_mkt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical hook kept for compatibility.

    The reproduction package starts from player1_all_categorized.xlsx, where TG
    Control/Market zero-sender beliefs are already present. No auxiliary data
    file is needed here.
    """
    return df


def fit_line_with_mean_ci(
    data: pd.DataFrame,
    x_col: str = "beliefs",
    y_col: str = "share_sent",
    x_grid: Iterable[float] = X_GRID,
) -> pd.DataFrame:
    """
    Fit y = a + b*x and return fitted mean and 95% confidence band.

    The band is the confidence interval for the fitted mean, not a prediction
    interval for individual observations.
    """
    clean = data[[x_col, y_col]].dropna().copy()
    if len(clean) < 3:
        raise ValueError("Not enough observations to fit a line.")

    x = clean[x_col].astype(float)
    y = clean[y_col].astype(float)

    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X)

    if COV_TYPE is None:
        result = model.fit()
    else:
        result = model.fit(cov_type=COV_TYPE)

    x_grid = np.asarray(list(x_grid), dtype=float)
    X_pred = sm.add_constant(x_grid, has_constant="add")
    pred = result.get_prediction(X_pred).summary_frame(alpha=0.05)

    return pd.DataFrame(
        {
            "x": x_grid,
            "fit": pred["mean"].to_numpy(dtype=float),
            "ci_low": pred["mean_ci_lower"].to_numpy(dtype=float),
            "ci_high": pred["mean_ci_upper"].to_numpy(dtype=float),
            "slope": float(result.params[x_col]),
            "intercept": float(result.params["const"]),
            "n": int(result.nobs),
        }
    )


def add_fitted_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    label: str,
    show_slope_in_legend: bool = True,
) -> None:
    """Fit one group and add its line plus mean-confidence band to an axis."""
    fitted = fit_line_with_mean_ci(data)
    slope = fitted["slope"].iloc[0]
    n = fitted["n"].iloc[0]

    if show_slope_in_legend:
        legend_label = f"{label} (slope={slope:.3f}, N={n})"
    else:
        legend_label = f"{label} (N={n})"

    line = ax.plot(
        fitted["x"],
        fitted["fit"],
        linewidth=2.6,
        label=legend_label,
    )[0]

    ax.fill_between(
        fitted["x"].to_numpy(dtype=float),
        fitted["ci_low"].to_numpy(dtype=float),
        fitted["ci_high"].to_numpy(dtype=float),
        color=line.get_color(),
        alpha=0.16,
        linewidth=0,
    )


def finish_graph(ax: plt.Axes, title: str) -> None:
    """Apply common formatting to the sensitivity graphs."""
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Beliefs", fontsize=12)
    ax.set_ylabel("Share Sent P1", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9.5)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_p1_categorized_data() -> pd.DataFrame:
    base = script_dir()
    input_path = base / P1_CATEGORIZED_INPUT

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input file: {input_path}\n"
            "Expected it at data/player1_all_categorized.xlsx "
            "relative to the script location."
        )

    df = pd.read_excel(input_path)
    df = normalize_columns(df)
    df = prepare_common_fields(df)

    for col in ["beliefs", "share_sent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "category_num" in df.columns:
        df["category_num"] = pd.to_numeric(df["category_num"], errors="coerce")

    df["category"] = df["category"].astype(str).str.strip()
    df = fill_tg_zero_beliefs_ctr_mkt(df)
    return df


def load_p1_all_data() -> pd.DataFrame:
    input_path = script_dir() / P1_INPUT
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_excel(input_path)
    df = prepare_common_fields(df)

    required = ["share_sent", "beliefs"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required Player 1 columns: {missing}")

    for col in ["share_sent", "beliefs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "beliefs_hp" in df.columns:
        df["beliefs_hp"] = pd.to_numeric(df["beliefs_hp"], errors="coerce")

    df = fill_tg_zero_beliefs_ctr_mkt(df)
    return df


def load_p2_all_data() -> pd.DataFrame:
    input_path = script_dir() / P2_INPUT
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_excel(input_path)
    df = prepare_common_fields(df)

    required = ["share_sent_p1", "share_sent", "choice"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required Player 2 columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "share_sent_hp" in df.columns:
        df["share_sent_hp"] = pd.to_numeric(df["share_sent_hp"], errors="coerce")
    if "choice_hp" in df.columns:
        df["choice_hp"] = pd.to_numeric(df["choice_hp"], errors="coerce")

    return df


def category_data_for_game(df: pd.DataFrame, game: str) -> pd.DataFrame:
    """
    Return Control observations for one game, keeping only categories whose
    within-game share is at least MIN_CATEGORY_SHARE.
    """
    d = df.loc[df["game"].eq(game) & df["story"].eq(0)].copy()

    if "category_num" in d.columns:
        d = d.loc[d["category_num"].ne(0)]

    d = d.dropna(subset=["beliefs", "share_sent", "category"])

    if d.empty:
        raise ValueError(f"No category data available for game={game}, story=0.")

    category_counts = d.groupby("category", dropna=False).size().rename("category_n")
    d = d.join(category_counts, on="category")
    d["category_share"] = d["category_n"] / len(d)
    d = d.loc[d["category_share"].ge(MIN_CATEGORY_SHARE)].copy()

    if d["category"].nunique() == 0:
        raise ValueError(f"No categories above {MIN_CATEGORY_SHARE:.0%} for game={game}.")

    return d


def market_data_for_game(df: pd.DataFrame, game: str) -> pd.DataFrame:
    """Return Market and Control observations for one game."""
    d = df.loc[df["game"].eq(game) & df["story"].isin([0, 1])].copy()
    d = d.dropna(subset=["beliefs", "share_sent", "story"])

    if d.empty:
        raise ValueError(f"No Market/Control data available for game={game}.")

    return d


def comparison_data_for_game(
    df: pd.DataFrame,
    game: str,
    story_values: list[int],
    comparison_label: str,
) -> pd.DataFrame:
    """Return observations for one beliefs-sensitivity comparison within a game."""
    d = df.loc[df["game"].eq(game) & df["story"].isin(story_values)].copy()
    d = d.dropna(subset=["beliefs", "share_sent", "story"])

    if d.empty:
        raise ValueError(f"No {comparison_label} data available for game={game}.")

    return d


# ---------------------------------------------------------------------
# Sensitivity graph functions
# ---------------------------------------------------------------------

def plot_control_by_category(df: pd.DataFrame, game: str, output_name: str) -> None:
    d = category_data_for_game(df, game)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    category_order = (
        d.groupby("category")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for category in category_order:
        add_fitted_line(ax, d.loc[d["category"].eq(category)], label=category)

    finish_graph(ax, f"{game.upper()} Control: Beliefs Sensitivity by category")
    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_market_vs_control(df: pd.DataFrame, game: str, output_name: str) -> None:
    d = market_data_for_game(df, game)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for story_value in [0, 1]:
        group = d.loc[d["story"].eq(story_value)]
        if group.empty:
            print(f"Warning: no observations for game={game}, story={story_value}")
            continue
        add_fitted_line(ax, group, label=STORY_LABELS[story_value])

    finish_graph(ax, f"{game.upper()}: Beliefs Sensitivity Market vs Control")
    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_story_comparison(
    df: pd.DataFrame,
    game: str,
    treated_story: int,
    baseline_story: int,
    comparison_title: str,
    output_name: str,
) -> None:
    """Plot one two-story beliefs sensitivity comparison for a game."""
    d = comparison_data_for_game(
        df,
        game=game,
        story_values=[baseline_story, treated_story],
        comparison_label=comparison_title,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for story_value in [baseline_story, treated_story]:
        group = d.loc[d["story"].eq(story_value)]
        if group.empty:
            print(f"Warning: no observations for game={game}, story={story_value}")
            continue
        add_fitted_line(ax, group, label=STORY_LABELS[story_value])

    finish_graph(ax, f"{game.upper()}: Beliefs Sensitivity {comparison_title}")
    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def player1_story_game_slice(p1: pd.DataFrame, game: str, story: int) -> pd.DataFrame:
    """Return one Player 1 story-by-game slice."""
    return p1.loc[p1["game"].eq(game) & p1["story"].eq(story)].copy()


def player2_story_game_slice(p2: pd.DataFrame, game: str, story: int) -> pd.DataFrame:
    """Return one Player 2 story-by-game slice."""
    return p2.loc[p2["game"].eq(game) & p2["story"].eq(story)].copy()


def tg_reference_share_from_p1(p1_slice: pd.DataFrame) -> pd.Series:
    """
    Return the TG action that the elicited belief refers to.

    For senders who chose 0, the elicited belief corresponds to the hypothetical
    case where they sent 1, i.e. share_sent = 1/6.
    """
    share_sent = pd.to_numeric(p1_slice["share_sent"], errors="coerce")
    reference_share = share_sent.where(~share_sent.eq(0), 1 / 6)
    return normalize_share_values(reference_share)


def actual_outcome_by_share(
    p2_slice: pd.DataFrame,
    game: str,
) -> pd.DataFrame:
    """Return realized average Player 2 behavior by P1 share within one cell."""
    outcome_col = "share_sent" if game == "tg" else "choice"
    benchmark = p2_slice[["share_sent_p1", outcome_col]].dropna().copy()
    benchmark["reference_share"] = normalize_share_values(benchmark["share_sent_p1"])
    benchmark = (
        benchmark.groupby("reference_share", as_index=False)[outcome_col]
        .mean()
        .rename(columns={outcome_col: "actual_outcome"})
    )
    return benchmark


def compute_signed_forecast_errors(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
) -> pd.Series:
    """Compute signed belief minus realized outcome for one game-treatment cell."""
    sender = p1_slice[["beliefs", "share_sent"]].dropna().copy()

    if game == "tg":
        sender["reference_share"] = tg_reference_share_from_p1(sender)
    else:
        sender["reference_share"] = normalize_share_values(sender["share_sent"])

    actual = actual_outcome_by_share(p2_slice, game=game)
    sender = sender.merge(actual, on="reference_share", how="left")
    sender = sender.dropna(subset=["beliefs", "actual_outcome"])

    return pd.to_numeric(sender["beliefs"], errors="coerce") - pd.to_numeric(
        sender["actual_outcome"], errors="coerce"
    )


def bootstrap_signed_forecast_error_stats(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean signed forecast error for one game-treatment cell."""
    forecast_errors = compute_signed_forecast_errors(p1_slice, p2_slice, game=game)

    if forecast_errors.empty:
        return {
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": 0,
            "benchmark_n": 0,
        }

    bootstrap_means = []
    rng = np.random.default_rng(seed)

    for _ in range(BOOTSTRAP_REPS):
        p1_boot = p1_slice.sample(
            n=len(p1_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p2_boot = p2_slice.sample(
            n=len(p2_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        boot_errors = compute_signed_forecast_errors(p1_boot, p2_boot, game=game)
        if not boot_errors.empty:
            bootstrap_means.append(float(boot_errors.mean()))

    if bootstrap_means:
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = np.nan
        ci_high = np.nan

    benchmark_n = len(actual_outcome_by_share(p2_slice, game=game))

    return {
        "mean": float(forecast_errors.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": int(len(forecast_errors)),
        "benchmark_n": int(benchmark_n),
    }


def compute_forecast_error_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean signed forecast error by treatment and game."""
    rows = []
    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]

    for story in story_order:
        for game in game_order:
            p1_slice = player1_story_game_slice(p1, game=game, story=story)
            p2_slice = player2_story_game_slice(p2, game=game, story=story)
            stats = bootstrap_signed_forecast_error_stats(
                p1_slice,
                p2_slice,
                game=game,
                seed=BOOTSTRAP_SEED + story * 100 + (0 if game == "ug" else 1),
            )
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def plot_forecast_error_by_treatment(summary: pd.DataFrame) -> None:
    """Plot one grouped bar chart of signed forecast error by treatment and game."""
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]
    x = np.arange(len(story_order))
    width = 0.34

    for idx, game in enumerate(game_order):
        subset = (
            summary[summary["game"].eq(game)]
            .set_index("story")
            .reindex(story_order)
        )
        positions = x + (idx - 0.5) * width
        centers = subset["mean_pp"].to_numpy()
        lower = centers - subset["ci_low_pp"].to_numpy()
        upper = subset["ci_high_pp"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=FORECAST_GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.7,
            label=game.upper(),
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([STORY_LABELS[story] for story in story_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("Forecast Error by Treatment and Game", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="upper right")

    save_figure(fig, script_dir() / FIGURES_DIR / "forecast_error_by_treatment.png")


def compute_ug_forecast_error_by_category_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute UG signed forecast error by category and treatment."""
    ug = p1.loc[p1["game"].eq("ug")].copy()
    rows = []

    observed_categories = ug["category"].dropna().astype(str).str.strip().unique().tolist()
    category_order = [
        category for category in UG_CATEGORY_ORDER if category in observed_categories
    ]
    category_order.extend(
        sorted(category for category in observed_categories if category not in category_order)
    )

    for category in category_order:
        for story in [0, 1, 2, 4]:
            p1_slice = ug.loc[
                ug["story"].eq(story) & ug["category"].eq(category)
            ].copy()
            p2_slice = player2_story_game_slice(p2, game="ug", story=story)
            stats = bootstrap_signed_forecast_error_stats(
                p1_slice,
                p2_slice,
                game="ug",
                seed=BOOTSTRAP_SEED + story * 100 + len(category),
            )
            rows.append(
                {
                    "category": category,
                    "category_label": UG_CATEGORY_LABELS.get(category, category),
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": "ug",
                    "game_label": "UG",
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def save_ug_forecast_error_by_category_summary(summary: pd.DataFrame) -> None:
    """Write the UG category-level forecast-error summary to CSV."""
    output_path = script_dir() / UG_CATEGORY_FORECAST_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_ug_forecast_error_by_category(summary: pd.DataFrame) -> None:
    """Plot UG signed forecast error by category with separate treatment bars."""
    fig, ax = plt.subplots(figsize=(11.2, 6.1))

    category_order = [
        category for category in UG_CATEGORY_ORDER if category in summary["category"].unique()
    ]
    x = np.arange(len(category_order))
    story_order = [0, 1, 2, 4]
    width = 0.19

    for idx, story in enumerate(story_order):
        subset = (
            summary.loc[summary["story"].eq(story)]
            .set_index("category")
            .reindex(category_order)
        )
        positions = x + (idx - 1.5) * width
        centers = subset["mean_pp"].to_numpy(dtype=float)
        lower = centers - subset["ci_low_pp"].to_numpy(dtype=float)
        upper = subset["ci_high_pp"].to_numpy(dtype=float) - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=BELIEF_TREATMENT_COLORS[story],
            edgecolor="black",
            linewidth=0.7,
            label=STORY_LABELS[story],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([UG_CATEGORY_LABELS.get(category, category) for category in category_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("UG Forecast Error by Category and Treatment", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    save_figure(fig, script_dir() / FIGURES_DIR / "ug_forecast_error_by_category.png")


def compute_tg_forecast_error_by_category_control_market_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute TG signed forecast error by category for Control and Market."""
    tg = p1.loc[p1["game"].eq("tg")].copy()
    rows = []
    story_order = [0, 1]

    observed_categories = tg["category"].dropna().astype(str).str.strip().unique().tolist()
    category_order = [
        category for category in UG_CATEGORY_ORDER if category in observed_categories
    ]
    category_order.extend(
        sorted(category for category in observed_categories if category not in category_order)
    )

    for category_idx, category in enumerate(category_order):
        for story in story_order:
            p1_slice = tg.loc[
                tg["story"].eq(story) & tg["category"].eq(category)
            ].copy()
            p2_slice = player2_story_game_slice(p2, game="tg", story=story)
            stats = bootstrap_signed_forecast_error_stats(
                p1_slice,
                p2_slice,
                game="tg",
                seed=BOOTSTRAP_SEED + 5000 + category_idx * 100 + story,
            )
            rows.append(
                {
                    "category": category,
                    "category_label": UG_CATEGORY_LABELS.get(category, category),
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": "tg",
                    "game_label": "TG",
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def save_tg_forecast_error_by_category_control_market_summary(summary: pd.DataFrame) -> None:
    """Write the TG Control/Market category-level forecast-error summary to CSV."""
    output_path = script_dir() / TG_CATEGORY_FORECAST_CM_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_tg_forecast_error_by_category_control_market(summary: pd.DataFrame) -> None:
    """Plot TG signed forecast error by category for Control and Market."""
    fig, ax = plt.subplots(figsize=(10.4, 6.0))

    category_order = [
        category for category in UG_CATEGORY_ORDER if category in summary["category"].unique()
    ]
    x = np.arange(len(category_order))
    story_order = [0, 1]
    width = 0.34

    for idx, story in enumerate(story_order):
        subset = (
            summary.loc[summary["story"].eq(story)]
            .set_index("category")
            .reindex(category_order)
        )
        positions = x + (idx - 0.5) * width
        centers = subset["mean_pp"].to_numpy(dtype=float)
        lower = centers - subset["ci_low_pp"].to_numpy(dtype=float)
        upper = subset["ci_high_pp"].to_numpy(dtype=float) - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=BELIEF_TREATMENT_COLORS[story],
            edgecolor="black",
            linewidth=0.7,
            label=STORY_LABELS[story],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([UG_CATEGORY_LABELS.get(category, category) for category in category_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("TG Forecast Error by Category: Control vs Market", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="upper right")

    save_figure(
        fig,
        script_dir() / FIGURES_DIR / "tg_forecast_error_by_category_control_market.png",
    )


def compute_tg_forecast_error_by_category_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute TG signed forecast error by category and treatment."""
    tg = p1.loc[p1["game"].eq("tg")].copy()
    rows = []
    story_order = [0, 1, 2, 4]

    observed_categories = tg["category"].dropna().astype(str).str.strip().unique().tolist()
    category_order = [
        category for category in UG_CATEGORY_ORDER if category in observed_categories
    ]
    category_order.extend(
        sorted(category for category in observed_categories if category not in category_order)
    )

    for category_idx, category in enumerate(category_order):
        for story in story_order:
            p1_slice = tg.loc[
                tg["story"].eq(story) & tg["category"].eq(category)
            ].copy()
            p2_slice = player2_story_game_slice(p2, game="tg", story=story)
            stats = bootstrap_signed_forecast_error_stats(
                p1_slice,
                p2_slice,
                game="tg",
                seed=BOOTSTRAP_SEED + 8000 + category_idx * 100 + story,
            )
            rows.append(
                {
                    "category": category,
                    "category_label": UG_CATEGORY_LABELS.get(category, category),
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": "tg",
                    "game_label": "TG",
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def save_tg_forecast_error_by_category_summary(summary: pd.DataFrame) -> None:
    """Write the TG category-level forecast-error summary to CSV."""
    output_path = script_dir() / TG_CATEGORY_FORECAST_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_tg_forecast_error_by_category(summary: pd.DataFrame) -> None:
    """Plot TG signed forecast error by category with separate treatment bars."""
    fig, ax = plt.subplots(figsize=(11.2, 6.1))

    category_order = [
        category for category in UG_CATEGORY_ORDER if category in summary["category"].unique()
    ]
    x = np.arange(len(category_order))
    story_order = [0, 1, 2, 4]
    width = 0.19

    for idx, story in enumerate(story_order):
        subset = (
            summary.loc[summary["story"].eq(story)]
            .set_index("category")
            .reindex(category_order)
        )
        positions = x + (idx - 1.5) * width
        centers = subset["mean_pp"].to_numpy(dtype=float)
        lower = centers - subset["ci_low_pp"].to_numpy(dtype=float)
        upper = subset["ci_high_pp"].to_numpy(dtype=float) - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=BELIEF_TREATMENT_COLORS[story],
            edgecolor="black",
            linewidth=0.7,
            label=STORY_LABELS[story],
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([UG_CATEGORY_LABELS.get(category, category) for category in category_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("TG Forecast Error by Category and Treatment", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    save_figure(fig, script_dir() / FIGURES_DIR / "tg_forecast_error_by_category.png")


def compute_signed_forecast_errors_hp_actual(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
) -> pd.Series:
    """Compute signed hypothetical forecast error using actual receiver behavior at share 1/3."""
    sender = p1_slice[["beliefs_hp"]].dropna().copy()
    if sender.empty:
        return pd.Series(dtype=float)

    actual = actual_outcome_by_share(p2_slice, game=game)
    actual_at_hp = actual.loc[actual["reference_share"].eq(HP_REFERENCE_SHARE), "actual_outcome"]

    if actual_at_hp.empty:
        return pd.Series(dtype=float)

    benchmark = float(actual_at_hp.iloc[0])
    beliefs_hp = pd.to_numeric(sender["beliefs_hp"], errors="coerce").dropna()
    return beliefs_hp - benchmark


def bootstrap_signed_forecast_error_hp_actual_stats(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean hypothetical signed forecast error using actual receiver behavior."""
    forecast_errors = compute_signed_forecast_errors_hp_actual(p1_slice, p2_slice, game=game)

    if forecast_errors.empty:
        return {
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": 0,
            "benchmark_n": 0,
        }

    bootstrap_means = []
    rng = np.random.default_rng(seed)

    for _ in range(BOOTSTRAP_REPS):
        p1_boot = p1_slice.sample(
            n=len(p1_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p2_boot = p2_slice.sample(
            n=len(p2_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        boot_errors = compute_signed_forecast_errors_hp_actual(p1_boot, p2_boot, game=game)
        if not boot_errors.empty:
            bootstrap_means.append(float(boot_errors.mean()))

    if bootstrap_means:
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = np.nan
        ci_high = np.nan

    benchmark = actual_outcome_by_share(p2_slice, game=game)
    benchmark_n = int(benchmark["reference_share"].eq(HP_REFERENCE_SHARE).sum())

    return {
        "mean": float(forecast_errors.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": int(len(forecast_errors)),
        "benchmark_n": benchmark_n,
    }


def compute_forecast_error_hp_actual_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean signed hypothetical forecast error by treatment and game using actual receiver behavior."""
    rows = []
    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]

    for story in story_order:
        for game in game_order:
            p1_slice = player1_story_game_slice(p1, game=game, story=story)
            p2_slice = player2_story_game_slice(p2, game=game, story=story)
            stats = bootstrap_signed_forecast_error_hp_actual_stats(
                p1_slice,
                p2_slice,
                game=game,
                seed=BOOTSTRAP_SEED + 2000 + story * 100 + (0 if game == "ug" else 1),
            )
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def plot_forecast_error_hp_actual_by_treatment(summary: pd.DataFrame) -> None:
    """Plot grouped bars for hypothetical forecast error using actual receiver behavior."""
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]
    x = np.arange(len(story_order))
    width = 0.34

    for idx, game in enumerate(game_order):
        subset = (
            summary[summary["game"].eq(game)]
            .set_index("story")
            .reindex(story_order)
        )
        positions = x + (idx - 0.5) * width
        centers = subset["mean_pp"].to_numpy()
        lower = centers - subset["ci_low_pp"].to_numpy()
        upper = subset["ci_high_pp"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=FORECAST_GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.7,
            label=game.upper(),
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([STORY_LABELS[story] for story in story_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("Hypothetical Forecast Error by Treatment and Game", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="upper right")

    save_figure(fig, script_dir() / FIGURES_DIR / "forecast_error_hp_actual_by_treatment.png")


def compute_signed_forecast_errors_hp(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
) -> pd.Series:
    """Compute signed hypothetical forecast error using Player 2 hypothetical responses."""
    sender = p1_slice[["beliefs_hp"]].dropna().copy()
    if sender.empty:
        return pd.Series(dtype=float)

    benchmark_col = "share_sent_hp" if game == "tg" else "choice_hp"
    receiver_hp = pd.to_numeric(p2_slice.get(benchmark_col), errors="coerce").dropna()
    if receiver_hp.empty:
        return pd.Series(dtype=float)

    benchmark = float(receiver_hp.mean())
    beliefs_hp = pd.to_numeric(sender["beliefs_hp"], errors="coerce").dropna()
    return beliefs_hp - benchmark


def bootstrap_signed_forecast_error_hp_stats(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean hypothetical signed forecast error for one game-treatment cell."""
    forecast_errors = compute_signed_forecast_errors_hp(p1_slice, p2_slice, game=game)

    if forecast_errors.empty:
        return {
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": 0,
            "benchmark_n": 0,
        }

    bootstrap_means = []
    rng = np.random.default_rng(seed)

    for _ in range(BOOTSTRAP_REPS):
        p1_boot = p1_slice.sample(
            n=len(p1_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p2_boot = p2_slice.sample(
            n=len(p2_slice),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        boot_errors = compute_signed_forecast_errors_hp(p1_boot, p2_boot, game=game)
        if not boot_errors.empty:
            bootstrap_means.append(float(boot_errors.mean()))

    if bootstrap_means:
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = np.nan
        ci_high = np.nan

    benchmark_col = "share_sent_hp" if game == "tg" else "choice_hp"
    benchmark_n = int(pd.to_numeric(p2_slice.get(benchmark_col), errors="coerce").notna().sum())

    return {
        "mean": float(forecast_errors.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": int(len(forecast_errors)),
        "benchmark_n": benchmark_n,
    }


def compute_forecast_error_hp_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean signed hypothetical forecast error by treatment and game."""
    rows = []
    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]

    for story in story_order:
        for game in game_order:
            p1_slice = player1_story_game_slice(p1, game=game, story=story)
            p2_slice = player2_story_game_slice(p2, game=game, story=story)
            stats = bootstrap_signed_forecast_error_hp_stats(
                p1_slice,
                p2_slice,
                game=game,
                seed=BOOTSTRAP_SEED + 1000 + story * 100 + (0 if game == "ug" else 1),
            )
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                    "benchmark_n": stats["benchmark_n"],
                }
            )

    return pd.DataFrame(rows)


def regression_constant_stats(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
) -> dict[str, float]:
    """Estimate the intercept from y on x and return its 95% CI."""
    clean = data[[y_col, x_col]].dropna().copy()
    n_obs = len(clean)

    if n_obs < 3:
        return {
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": n_obs,
        }

    X = sm.add_constant(clean[x_col].astype(float), has_constant="add")
    y = clean[y_col].astype(float)
    model = sm.OLS(y, X)

    if COV_TYPE is None:
        result = model.fit()
    else:
        result = model.fit(cov_type=COV_TYPE)

    intercept = float(result.params["const"])
    ci = result.conf_int(alpha=0.05)
    ci_low = float(ci.loc["const", 0])
    ci_high = float(ci.loc["const", 1])

    return {
        "mean": intercept,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": int(result.nobs),
    }


def compute_beliefs_intercept_summary(p1: pd.DataFrame) -> pd.DataFrame:
    """Compute treatment-specific regression intercepts from beliefs on share_sent."""
    rows = []

    for story in [0, 1, 2, 4]:
        for game in ["ug", "tg"]:
            p1_slice = (
                player1_story_game_slice(p1, game=game, story=story)
                .dropna(subset=["beliefs", "share_sent"])
                .copy()
            )
            stats = regression_constant_stats(
                p1_slice,
                y_col="beliefs",
                x_col="share_sent",
            )
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                }
            )

    return pd.DataFrame(rows)


def compute_beliefs_hp_mean_summary(p1: pd.DataFrame) -> pd.DataFrame:
    """Compute treatment-specific means of hypothetical beliefs."""
    rows = []

    for story in [0, 1, 2, 4]:
        for game in ["ug", "tg"]:
            p1_slice = player1_story_game_slice(p1, game=game, story=story)
            stats = mean_ci(p1_slice["beliefs_hp"])
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": int(stats["n"]),
                }
            )

    return pd.DataFrame(rows)


def compute_beliefs_mean_summary(p1: pd.DataFrame) -> pd.DataFrame:
    """Compute treatment-specific means of actual beliefs."""
    rows = []

    for story in [0, 1, 2, 4]:
        for game in ["ug", "tg"]:
            p1_slice = player1_story_game_slice(p1, game=game, story=story)
            stats = mean_ci(p1_slice["beliefs"])
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": int(stats["n"]),
                }
            )

    return pd.DataFrame(rows)


def chosen_share_weights_from_control(p1_control: pd.DataFrame) -> pd.DataFrame:
    """Return control-group weights by chosen share_sent."""
    weights = p1_control[["share_sent"]].dropna().copy()
    weights["chosen_share"] = normalize_share_values(weights["share_sent"])
    weights = (
        weights.groupby("chosen_share", as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    total = weights["n"].sum()
    weights["weight"] = weights["n"] / total if total else np.nan
    return weights[["chosen_share", "weight"]]


def chosen_share_weights_from_baseline(p1_baseline: pd.DataFrame) -> pd.DataFrame:
    """Return baseline-group weights by chosen share_sent."""
    return chosen_share_weights_from_control(p1_baseline)


def weighted_beliefs_mean_from_control_shares(
    p1_target: pd.DataFrame,
    control_weights: pd.DataFrame,
) -> float:
    """Compute average beliefs in a target cell using Control chosen-share weights."""
    target = p1_target[["beliefs", "share_sent"]].dropna().copy()
    if target.empty or control_weights.empty:
        return np.nan

    target["chosen_share"] = normalize_share_values(target["share_sent"])
    conditional = (
        target.groupby("chosen_share", as_index=False)["beliefs"]
        .mean()
        .rename(columns={"beliefs": "belief_mean"})
    )

    weighted = control_weights.merge(conditional, on="chosen_share", how="left")
    weighted = weighted.dropna(subset=["belief_mean"]).copy()
    if weighted.empty:
        return np.nan
    weight_total = weighted["weight"].sum()
    if weight_total <= 0:
        return np.nan
    weighted["weight"] = weighted["weight"] / weight_total

    return float((weighted["weight"] * weighted["belief_mean"]).sum())


def forecast_errors_with_chosen_share(
    p1_slice: pd.DataFrame,
    p2_slice: pd.DataFrame,
    game: str,
) -> pd.DataFrame:
    """Return participant-level signed forecast errors with chosen share attached."""
    sender = p1_slice[["beliefs", "share_sent"]].dropna().copy()
    if sender.empty:
        return pd.DataFrame(columns=["chosen_share", "forecast_error"])

    sender["chosen_share"] = normalize_share_values(sender["share_sent"])

    if game == "tg":
        sender["reference_share"] = tg_reference_share_from_p1(sender)
    else:
        sender["reference_share"] = normalize_share_values(sender["share_sent"])

    actual = actual_outcome_by_share(p2_slice, game=game)
    sender = sender.merge(actual, on="reference_share", how="left")
    sender = sender.dropna(subset=["beliefs", "actual_outcome"])
    sender["forecast_error"] = pd.to_numeric(sender["beliefs"], errors="coerce") - pd.to_numeric(
        sender["actual_outcome"], errors="coerce"
    )

    return sender[["chosen_share", "forecast_error"]]


def weighted_forecast_error_from_control_shares(
    p1_target: pd.DataFrame,
    p2_target: pd.DataFrame,
    game: str,
    control_weights: pd.DataFrame,
) -> float:
    """Compute mean forecast error in a target cell using Control chosen-share weights."""
    if control_weights.empty:
        return np.nan

    target_errors = forecast_errors_with_chosen_share(p1_target, p2_target, game=game)
    if target_errors.empty:
        return np.nan

    conditional = (
        target_errors.groupby("chosen_share", as_index=False)["forecast_error"]
        .mean()
        .rename(columns={"forecast_error": "forecast_error_mean"})
    )

    weighted = control_weights.merge(conditional, on="chosen_share", how="left")
    weighted = weighted.dropna(subset=["forecast_error_mean"]).copy()
    if weighted.empty:
        return np.nan
    weight_total = weighted["weight"].sum()
    if weight_total <= 0:
        return np.nan
    weighted["weight"] = weighted["weight"] / weight_total

    return float((weighted["weight"] * weighted["forecast_error_mean"]).sum())


def weighted_p2_outcome_from_baseline_shares(
    p2_target: pd.DataFrame,
    game: str,
    baseline_weights: pd.DataFrame,
) -> float:
    """Compute average actual Player 2 outcome using baseline chosen-share weights."""
    outcome_col = "share_sent" if game == "tg" else "choice"
    target = p2_target[["share_sent_p1", outcome_col]].dropna().copy()
    if target.empty or baseline_weights.empty:
        return np.nan

    target["chosen_share"] = normalize_share_values(target["share_sent_p1"])
    conditional = (
        target.groupby("chosen_share", as_index=False)[outcome_col]
        .mean()
        .rename(columns={outcome_col: "p2_outcome_mean"})
    )

    weighted = baseline_weights.merge(conditional, on="chosen_share", how="left")
    weighted = weighted.dropna(subset=["p2_outcome_mean"]).copy()
    if weighted.empty:
        return np.nan
    weight_total = weighted["weight"].sum()
    if weight_total <= 0:
        return np.nan
    weighted["weight"] = weighted["weight"] / weight_total

    return float((weighted["weight"] * weighted["p2_outcome_mean"]).sum())


def bootstrap_market_vs_control_reweighted_stats(
    p1_control: pd.DataFrame,
    p1_target: pd.DataFrame,
    p2_target: pd.DataFrame,
    game: str,
    story: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap control-reweighted beliefs and forecast error for one game-treatment cell."""
    control_weights = chosen_share_weights_from_control(p1_control)
    belief_mean = weighted_beliefs_mean_from_control_shares(p1_target, control_weights)
    forecast_error_mean = weighted_forecast_error_from_control_shares(
        p1_target,
        p2_target,
        game=game,
        control_weights=control_weights,
    )

    if pd.isna(belief_mean) or pd.isna(forecast_error_mean):
        return {
            "belief_mean": np.nan,
            "belief_ci_low": np.nan,
            "belief_ci_high": np.nan,
            "forecast_mean": np.nan,
            "forecast_ci_low": np.nan,
            "forecast_ci_high": np.nan,
            "n_obs": 0,
        }

    rng = np.random.default_rng(seed)
    belief_boot = []
    forecast_boot = []

    for _ in range(BOOTSTRAP_REPS):
        p1_control_boot = p1_control.sample(
            n=len(p1_control),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p1_target_boot = p1_target.sample(
            n=len(p1_target),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p2_target_boot = p2_target.sample(
            n=len(p2_target),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )

        control_weights_boot = chosen_share_weights_from_control(p1_control_boot)
        belief_boot_mean = weighted_beliefs_mean_from_control_shares(
            p1_target_boot,
            control_weights_boot,
        )
        forecast_boot_mean = weighted_forecast_error_from_control_shares(
            p1_target_boot,
            p2_target_boot,
            game=game,
            control_weights=control_weights_boot,
        )

        if pd.notna(belief_boot_mean):
            belief_boot.append(float(belief_boot_mean))
        if pd.notna(forecast_boot_mean):
            forecast_boot.append(float(forecast_boot_mean))

    if belief_boot:
        belief_ci_low, belief_ci_high = np.quantile(belief_boot, [0.025, 0.975])
    else:
        belief_ci_low = np.nan
        belief_ci_high = np.nan

    if forecast_boot:
        forecast_ci_low, forecast_ci_high = np.quantile(forecast_boot, [0.025, 0.975])
    else:
        forecast_ci_low = np.nan
        forecast_ci_high = np.nan

    return {
        "belief_mean": float(belief_mean),
        "belief_ci_low": float(belief_ci_low),
        "belief_ci_high": float(belief_ci_high),
        "forecast_mean": float(forecast_error_mean),
        "forecast_ci_low": float(forecast_ci_low),
        "forecast_ci_high": float(forecast_ci_high),
        "n_obs": int(len(p1_target)),
    }


def compute_market_vs_control_reweighted_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Control-weighted beliefs and forecast errors for Control and Market."""
    belief_rows = []
    forecast_rows = []

    for game in ["ug", "tg"]:
        p1_control = player1_story_game_slice(p1, game=game, story=0)

        for story in [0, 1]:
            p1_target = player1_story_game_slice(p1, game=game, story=story)
            p2_target = player2_story_game_slice(p2, game=game, story=story)
            stats = bootstrap_market_vs_control_reweighted_stats(
                p1_control=p1_control,
                p1_target=p1_target,
                p2_target=p2_target,
                game=game,
                story=story,
                seed=BOOTSTRAP_SEED + 4000 + story * 100 + (0 if game == "ug" else 1),
            )

            belief_rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["belief_mean"],
                    "ci_low": stats["belief_ci_low"],
                    "ci_high": stats["belief_ci_high"],
                    "mean_pp": stats["belief_mean"] * 100 if pd.notna(stats["belief_mean"]) else np.nan,
                    "ci_low_pp": stats["belief_ci_low"] * 100 if pd.notna(stats["belief_ci_low"]) else np.nan,
                    "ci_high_pp": stats["belief_ci_high"] * 100 if pd.notna(stats["belief_ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                }
            )

            forecast_rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["forecast_mean"],
                    "ci_low": stats["forecast_ci_low"],
                    "ci_high": stats["forecast_ci_high"],
                    "mean_pp": stats["forecast_mean"] * 100 if pd.notna(stats["forecast_mean"]) else np.nan,
                    "ci_low_pp": stats["forecast_ci_low"] * 100 if pd.notna(stats["forecast_ci_low"]) else np.nan,
                    "ci_high_pp": stats["forecast_ci_high"] * 100 if pd.notna(stats["forecast_ci_high"]) else np.nan,
                    "n_obs": stats["n_obs"],
                }
            )

    return pd.DataFrame(belief_rows), pd.DataFrame(forecast_rows)


def compute_p2_hypothetical_mean_summary(p2: pd.DataFrame) -> pd.DataFrame:
    """Compute treatment-specific means of Player 2 hypothetical outcomes."""
    rows = []

    for story in [0, 1, 2, 4]:
        for game in ["ug", "tg"]:
            p2_slice = player2_story_game_slice(p2, game=game, story=story)
            outcome_col = "share_sent_hp" if game == "tg" else "choice_hp"
            stats = mean_ci(p2_slice[outcome_col])
            rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "mean_pp": stats["mean"] * 100 if pd.notna(stats["mean"]) else np.nan,
                    "ci_low_pp": stats["ci_low"] * 100 if pd.notna(stats["ci_low"]) else np.nan,
                    "ci_high_pp": stats["ci_high"] * 100 if pd.notna(stats["ci_high"]) else np.nan,
                    "n_obs": int(stats["n"]),
                }
            )

    return pd.DataFrame(rows)


def bootstrap_reweighted_beliefs_and_p2_stats(
    p1_baseline: pd.DataFrame,
    p1_target: pd.DataFrame,
    p2_target: pd.DataFrame,
    game: str,
    seed: int,
) -> dict[str, float]:
    """Bootstrap baseline-weighted actual beliefs and P2 outcomes for one cell."""
    baseline_weights = chosen_share_weights_from_baseline(p1_baseline)
    belief_mean = weighted_beliefs_mean_from_control_shares(p1_target, baseline_weights)
    p2_mean = weighted_p2_outcome_from_baseline_shares(p2_target, game, baseline_weights)

    if pd.isna(belief_mean) or pd.isna(p2_mean):
        return {
            "belief_mean": np.nan,
            "belief_ci_low": np.nan,
            "belief_ci_high": np.nan,
            "p2_mean": np.nan,
            "p2_ci_low": np.nan,
            "p2_ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    belief_boot = []
    p2_boot = []

    for _ in range(BOOTSTRAP_REPS):
        p1_baseline_boot = p1_baseline.sample(
            n=len(p1_baseline),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p1_target_boot = p1_target.sample(
            n=len(p1_target),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        p2_target_boot = p2_target.sample(
            n=len(p2_target),
            replace=True,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )

        baseline_weights_boot = chosen_share_weights_from_baseline(p1_baseline_boot)
        belief_boot_mean = weighted_beliefs_mean_from_control_shares(
            p1_target_boot,
            baseline_weights_boot,
        )
        p2_boot_mean = weighted_p2_outcome_from_baseline_shares(
            p2_target_boot,
            game,
            baseline_weights_boot,
        )

        if pd.notna(belief_boot_mean):
            belief_boot.append(float(belief_boot_mean))
        if pd.notna(p2_boot_mean):
            p2_boot.append(float(p2_boot_mean))

    if belief_boot:
        belief_ci_low, belief_ci_high = np.quantile(belief_boot, [0.025, 0.975])
    else:
        belief_ci_low = np.nan
        belief_ci_high = np.nan

    if p2_boot:
        p2_ci_low, p2_ci_high = np.quantile(p2_boot, [0.025, 0.975])
    else:
        p2_ci_low = np.nan
        p2_ci_high = np.nan

    return {
        "belief_mean": float(belief_mean),
        "belief_ci_low": float(belief_ci_low),
        "belief_ci_high": float(belief_ci_high),
        "p2_mean": float(p2_mean),
        "p2_ci_low": float(p2_ci_low),
        "p2_ci_high": float(p2_ci_high),
    }


def compute_reweighted_beliefs_and_p2_outcomes_summary(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    baseline_story: int,
    treated_story: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute baseline-weighted actual beliefs and actual P2 outcomes for one comparison."""
    belief_rows = []
    p2_rows = []

    for game in ["ug", "tg"]:
        p1_baseline = player1_story_game_slice(p1, game=game, story=baseline_story)

        for story in [baseline_story, treated_story]:
            p1_target = player1_story_game_slice(p1, game=game, story=story)
            p2_target = player2_story_game_slice(p2, game=game, story=story)
            stats = bootstrap_reweighted_beliefs_and_p2_stats(
                p1_baseline=p1_baseline,
                p1_target=p1_target,
                p2_target=p2_target,
                game=game,
                seed=BOOTSTRAP_SEED + 6000 + baseline_story * 100 + story * 10 + (0 if game == "ug" else 1),
            )

            belief_rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["belief_mean"],
                    "ci_low": stats["belief_ci_low"],
                    "ci_high": stats["belief_ci_high"],
                    "mean_pp": stats["belief_mean"] * 100 if pd.notna(stats["belief_mean"]) else np.nan,
                    "ci_low_pp": stats["belief_ci_low"] * 100 if pd.notna(stats["belief_ci_low"]) else np.nan,
                    "ci_high_pp": stats["belief_ci_high"] * 100 if pd.notna(stats["belief_ci_high"]) else np.nan,
                    "n_obs": int(len(p1_target)),
                }
            )

            p2_rows.append(
                {
                    "story": story,
                    "story_label": STORY_LABELS[story],
                    "game": game,
                    "game_label": game.upper(),
                    "mean": stats["p2_mean"],
                    "ci_low": stats["p2_ci_low"],
                    "ci_high": stats["p2_ci_high"],
                    "mean_pp": stats["p2_mean"] * 100 if pd.notna(stats["p2_mean"]) else np.nan,
                    "ci_low_pp": stats["p2_ci_low"] * 100 if pd.notna(stats["p2_ci_low"]) else np.nan,
                    "ci_high_pp": stats["p2_ci_high"] * 100 if pd.notna(stats["p2_ci_high"]) else np.nan,
                    "n_obs": int(len(p2_target)),
                }
            )

    return pd.DataFrame(belief_rows), pd.DataFrame(p2_rows)


def shade_for_story(story: int) -> float:
    """Use lighter shades for baseline stories and darker shades for treated stories."""
    return {
        0: 0.78,  # Control
        1: 1.00,  # Market
        2: 0.78,  # Bonus
        4: 1.00,  # Aid
    }[story]


def scaled_color(hex_color: str, scale: float) -> tuple[float, float, float]:
    """Scale a base color toward white for within-game treatment comparisons."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(1 - (1 - rgb) * scale)


def combined_bar_positions(n_stories: int) -> tuple[np.ndarray, float]:
    """Return symmetric x positions for one metric cluster."""
    if n_stories == 1:
        return np.array([-0.22, 0.22], dtype=float), 0.38

    return np.array([-0.48, -0.16, 0.16, 0.48], dtype=float), 0.24


def plot_combined_beliefs_forecast_figure(
    beliefs_summary: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    stories: list[int],
    title: str,
    output_name: str,
) -> None:
    """Plot beliefs and forecast error side by side for a given treatment comparison."""
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    game_order = ["ug", "tg"]
    beliefs_center = 0.0
    forecast_center = 1.55 if len(stories) == 1 else 2.55
    positions_template, width = combined_bar_positions(len(stories))

    legend_handles = {}

    for metric_name, summary, cluster_center in [
        ("beliefs", beliefs_summary, beliefs_center),
        ("forecast", forecast_summary, forecast_center),
    ]:
        positions = cluster_center + positions_template
        bar_idx = 0
        tick_positions = []
        tick_labels = []

        for game in game_order:
            for story in stories:
                row = (
                    summary.loc[
                        summary["game"].eq(game) & summary["story"].eq(story)
                    ]
                    .head(1)
                )
                if row.empty:
                    bar_idx += 1
                    continue

                row = row.iloc[0]
                color = scaled_color(COMBINED_GAME_COLORS[game], shade_for_story(story))
                x_pos = positions[bar_idx]
                center = float(row["mean_pp"])
                lower = center - float(row["ci_low_pp"])
                upper = float(row["ci_high_pp"]) - center

                bars = ax.bar(
                    x_pos,
                    center,
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.errorbar(
                    x_pos,
                    center,
                    yerr=[[lower], [upper]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.9,
                    capsize=3,
                    zorder=4,
                )

                legend_label = (
                    game.upper()
                    if len(stories) == 1
                    else f"{game.upper()} {STORY_LABELS[story]}"
                )
                if legend_label not in legend_handles:
                    legend_handles[legend_label] = bars[0]

                tick_positions.append(x_pos)
                if len(stories) == 1:
                    tick_labels.append(game.upper())
                else:
                    tick_labels.append(f"{game.upper()}\n{STORY_LABELS[story]}")

                bar_idx += 1

        if metric_name == "beliefs":
            beliefs_ticks = tick_positions
            beliefs_labels = tick_labels
        else:
            forecast_ticks = tick_positions
            forecast_labels = tick_labels

    all_ticks = beliefs_ticks + forecast_ticks
    all_labels = beliefs_labels + forecast_labels
    all_positions = np.array(all_ticks, dtype=float)

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.axvline((beliefs_center + forecast_center) / 2, color="#BBBBBB", linewidth=1.1, zorder=1)
    ax.set_xlim(all_positions.min() - 0.42, all_positions.max() + 0.42)
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("Percentage points")
    ax.set_title(title, pad=30, loc="center")
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        beliefs_center,
        0.94,
        "Beliefs",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        forecast_center,
        0.94,
        "Forecast Error",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )
    ax.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        frameon=False,
        ncol=min(len(legend_handles), 4),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
    )

    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_beliefs_and_p2_outcomes_comparison(
    beliefs_summary: pd.DataFrame,
    p2_summary: pd.DataFrame,
    baseline_story: int,
    treated_story: int,
    title: str,
    output_name: str,
    p2_panel_title: str,
) -> None:
    """Plot one comparison with beliefs and Player 2 outcomes in side-by-side panels."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8), sharey=True)
    game_order = ["ug", "tg"]
    story_order = [baseline_story, treated_story]
    width = 0.34
    legend_handles = {}

    for ax, summary, panel_title in zip(
        axes,
        [beliefs_summary, p2_summary],
        ["Player 1 Beliefs", p2_panel_title],
    ):
        x = np.arange(len(game_order))

        for idx, story in enumerate(story_order):
            subset = (
                summary.loc[
                    summary["story"].eq(story) & summary["game"].isin(game_order)
                ]
                .set_index("game")
                .reindex(game_order)
            )
            positions = x + (idx - 0.5) * width
            centers = subset["mean_pp"].to_numpy(dtype=float)
            lower = centers - subset["ci_low_pp"].to_numpy(dtype=float)
            upper = subset["ci_high_pp"].to_numpy(dtype=float) - centers
            colors = [
                scaled_color(COMBINED_GAME_COLORS[game], shade_for_story(story))
                for game in game_order
            ]

            ax.bar(
                positions,
                centers,
                width=width * 0.94,
                color=colors,
                edgecolor="black",
                linewidth=0.7,
                label=STORY_LABELS[story],
                zorder=3,
            )
            ax.errorbar(
                positions,
                centers,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=3,
                zorder=4,
            )

            for pos_idx, game in enumerate(game_order):
                legend_label = f"{game.upper()} {STORY_LABELS[story]}"
                if legend_label not in legend_handles:
                    legend_handles[legend_label] = Patch(
                        facecolor=colors[pos_idx],
                        edgecolor="black",
                        linewidth=0.7,
                        label=legend_label,
                    )

        ax.axhline(0, color="#666666", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([game.upper() for game in game_order])
        ax.set_title(panel_title, fontsize=12.5, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))

    axes[0].set_ylabel("Percentage points")
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )

    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_belief_treatment_summary(
    summary: pd.DataFrame,
    output_name: str,
    title: str,
    y_label: str,
) -> None:
    """Plot two treatment comparisons with UG/TG bars and 95% CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), sharey=True)
    game_order = ["ug", "tg"]
    width = 0.34

    for ax, comparison in zip(axes, BELIEF_TREATMENT_COMPARISONS):
        story_order = comparison["stories"]
        x = np.arange(len(game_order))

        for idx, story in enumerate(story_order):
            subset = (
                summary.loc[
                    summary["story"].eq(story) & summary["game"].isin(game_order)
                ]
                .set_index("game")
                .reindex(game_order)
            )
            positions = x + (idx - 0.5) * width
            centers = subset["mean_pp"].to_numpy(dtype=float)
            lower = centers - subset["ci_low_pp"].to_numpy(dtype=float)
            upper = subset["ci_high_pp"].to_numpy(dtype=float) - centers

            ax.bar(
                positions,
                centers,
                width=width * 0.94,
                color=BELIEF_TREATMENT_COLORS[story],
                edgecolor="black",
                linewidth=0.7,
                label=STORY_LABELS[story],
                zorder=3,
            )
            ax.errorbar(
                positions,
                centers,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=3,
                zorder=4,
            )

        ax.set_title(comparison["title"], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([game.upper() for game in game_order])
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
        ax.legend(frameon=False, ncol=1, loc="upper right")

    axes[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=14, y=0.98)

    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_forecast_error_hp_by_treatment(summary: pd.DataFrame) -> None:
    """Plot one grouped bar chart of hypothetical signed forecast error by treatment and game."""
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    story_order = [0, 1, 2, 4]
    game_order = ["ug", "tg"]
    x = np.arange(len(story_order))
    width = 0.34

    for idx, game in enumerate(game_order):
        subset = (
            summary[summary["game"].eq(game)]
            .set_index("story")
            .reindex(story_order)
        )
        positions = x + (idx - 0.5) * width
        centers = subset["mean_pp"].to_numpy()
        lower = centers - subset["ci_low_pp"].to_numpy()
        upper = subset["ci_high_pp"].to_numpy() - centers

        ax.bar(
            positions,
            centers,
            width=width * 0.94,
            color=FORECAST_GAME_COLORS[game],
            edgecolor="black",
            linewidth=0.7,
            label=game.upper(),
            zorder=3,
        )
        ax.errorbar(
            positions,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.axhline(0, color="#666666", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([STORY_LABELS[story] for story in story_order])
    ax.set_ylabel("Mean signed forecast error (pp)")
    ax.set_title("Hypothetical Forecast Error by Treatment and Game", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(pp_formatter))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="upper right")

    save_figure(fig, script_dir() / FIGURES_DIR / "forecast_error_hp_by_treatment.png")


# ---------------------------------------------------------------------
# Two-panel figure functions
# ---------------------------------------------------------------------

def make_two_panel_inputs(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    game: str,
    story: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect the grouped data needed for one two-panel figure."""
    p1_slice = p1.loc[p1["game"].eq(game) & p1["story"].eq(story)].copy()
    p2_slice = p2.loc[p2["game"].eq(game) & p2["story"].eq(story)].copy()

    if p1_slice.empty:
        raise ValueError(f"No Player 1 observations for game={game}, story={story}.")
    if p2_slice.empty:
        raise ValueError(f"No Player 2 observations for game={game}, story={story}.")

    beliefs_summary = grouped_mean_ci(p1_slice, x_col="share_sent", y_col="beliefs")

    p2_outcome_col = "share_sent" if game == "tg" else "choice"
    p2_outcome_label = "Average Share Sent P2" if game == "tg" else "Acceptance Rate"
    p2_summary = grouped_mean_ci(p2_slice, x_col="share_sent_p1", y_col=p2_outcome_col)
    p2_summary["series_label"] = p2_outcome_label

    if game == "tg":
        counterfactual_benchmark = p2_summary.loc[
            p2_summary["x"].eq(round(1 / 6, SHARE_DECIMALS))
        ].copy()
        if not counterfactual_benchmark.empty:
            counterfactual_benchmark["x"] = 0.0
            p2_summary = (
                pd.concat([counterfactual_benchmark, p2_summary], ignore_index=True)
                .sort_values("x")
                .reset_index(drop=True)
            )

    distribution = (
        p1_slice["share_sent"]
        .dropna()
        .value_counts(normalize=True)
        .sort_index()
        .rename("share")
        .reset_index()
    )
    distribution.columns = ["x", "share"]
    distribution["share"] = distribution["share"] * 100

    return beliefs_summary, p2_summary, distribution


def style_two_panel_axes(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    game: str,
    top_ylabel: str,
) -> None:
    """Apply shared styling for the two-panel figures."""
    tick_grid = share_tick_grid(game)
    tick_labels = [f"{int(round(x * 100))}%" for x in tick_grid]

    for ax in [ax_top, ax_bottom]:
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(tick_grid)
        ax.set_xticklabels(tick_labels)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    ax_top.set_ylim(0, 1)
    ax_top.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_top.set_ylabel(top_ylabel)
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_bottom.set_ylabel("Share of participants (%)")
    ax_bottom.set_xlabel("Share Sent P1")
    ax_bottom.axvline(0.5, color="#1f77b4", linestyle=":", linewidth=2.5, alpha=0.9)


def plot_two_panel_figure(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    game: str,
    story: int,
) -> None:
    """Generate one story-by-game two-panel figure."""
    story_label = STORY_LABELS[story]
    story_slug = STORY_SLUGS[story]

    beliefs_summary, p2_summary, distribution = make_two_panel_inputs(
        p1=p1,
        p2=p2,
        game=game,
        story=story,
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=TWO_PANEL_FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.12},
    )

    p2_label = p2_summary["series_label"].iloc[0]
    title_prefix = (
        "Average Share Sent P2 and Beliefs"
        if game == "tg"
        else "Acceptance Rate and Beliefs"
    )
    ax_top.errorbar(
        p2_summary["x"],
        p2_summary["mean"],
        yerr=[
            p2_summary["mean"] - p2_summary["ci_low"],
            p2_summary["ci_high"] - p2_summary["mean"],
        ],
        fmt="o-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        capsize=4,
        label=p2_label,
    )
    ax_top.errorbar(
        beliefs_summary["x"],
        beliefs_summary["mean"],
        yerr=[
            beliefs_summary["mean"] - beliefs_summary["ci_low"],
            beliefs_summary["ci_high"] - beliefs_summary["mean"],
        ],
        fmt="s--",
        color="black",
        linewidth=1.8,
        markersize=5.5,
        capsize=4,
        label="Beliefs",
    )
    ax_top.legend(frameon=False, loc="center right")
    ax_top.set_title(
        f"{title_prefix} (top) and Share Sent P1 Distribution (bottom) - {story_label}",
        fontsize=18,
        pad=14,
    )

    bar_width = (1 / 24) * 0.62 if game == "ug" else (1 / 12) * 0.62
    ax_bottom.bar(
        distribution["x"],
        distribution["share"],
        width=bar_width,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.82,
    )

    top_ylabel = (
        "Average Share Sent P2 / Beliefs"
        if game == "tg"
        else "Acceptance Rate / Beliefs"
    )
    style_two_panel_axes(ax_top, ax_bottom, game=game, top_ylabel=top_ylabel)

    output_name = f"{game}_two_panel_{story_slug}_with_beliefs.png"
    save_figure(fig, script_dir() / FIGURES_DIR / output_name)


def plot_all_two_panel_figures(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    """Generate all story-by-game two-panel figures."""
    for game in ["tg", "ug"]:
        for story in [0, 1, 2, 4]:
            plot_two_panel_figure(p1=p1, p2=p2, game=game, story=story)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def write_figure_stats(
    forecast_error_summary: pd.DataFrame,
    forecast_error_hp_summary: pd.DataFrame,
) -> None:
    """Numeric log of the statistics rendered on the two paper figures, so prose
    numbers citing them can be audited without reading pixels (added 2026-07-16).
    The frames are the exact summaries the plotting functions receive."""
    lines: list[str] = []

    def add(title: str, frame: pd.DataFrame) -> None:
        lines.append(f"=== {title} ===")
        lines.append(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines.append("")

    add(
        "Signed forecast errors by treatment, chosen-action beliefs (fig forecast_error_by_treatment)",
        forecast_error_summary,
    )
    add(
        "Signed forecast errors by treatment, hypothetical beliefs at the reference action (fig forecast_error_hp_by_treatment)",
        forecast_error_hp_summary,
    )

    output_path = script_dir() / "output" / "tables" / "forecast_error_figure_stats.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


def main() -> None:
    p1_all = load_p1_all_data()
    p2_all = load_p2_all_data()

    forecast_error_summary = compute_forecast_error_summary(p1=p1_all, p2=p2_all)
    plot_forecast_error_by_treatment(forecast_error_summary)

    forecast_error_hp_summary = compute_forecast_error_hp_summary(p1=p1_all, p2=p2_all)
    plot_forecast_error_hp_by_treatment(forecast_error_hp_summary)

    write_figure_stats(forecast_error_summary, forecast_error_hp_summary)

if __name__ == "__main__":
    main()
