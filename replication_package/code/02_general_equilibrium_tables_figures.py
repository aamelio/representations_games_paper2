from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# These figures previously used the matplotlib defaults, which left them in a
# sans-serif face (unlike every other figure in the paper) and saved at 100 dpi,
# i.e. ~127 dpi once scaled into the page. FONT_SCALE plays the same role as in
# script 01: drawn ~6.5in wide, included at ~4.8in.
FONT_SCALE = 1.15

plt.rcParams.update(
    {
        "savefig.dpi": 300,
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # replication_package/
P1_PATH = PROJECT_ROOT / "data" / "player1_all_categorized.xlsx"
P2_PATH = PROJECT_ROOT / "data" / "player2_all_categorized.xlsx"
FIG_DIR = PROJECT_ROOT / "output" / "figures" / "fitted_fullpooled"
TEX_DIR = PROJECT_ROOT / "output" / "tables"
BOOTSTRAP_REPS = 400
BOOTSTRAP_SEED = 20260611

GAME_COLORS = {
    "dgkw": "#4C566A",
    "tg": "#8FBCBB",
    "ug": "#D08770",
}

GAME_COLORS_BRIGHT = {
    "dgkw": "#7B879D",
    "tg": "#BFE3E0",
    "ug": "#F2B39D",
}

COMPARISON_COLORS = {
    "aid_vs_bonus": "#1F4E79",
    "market_vs_control": "#C65D3A",
}

COMPARISONS = [
    {
        "slug": "aid_vs_bonus",
        "title": "Aid vs Bonus",
        "baseline_story": 2,
        "treated_story": 4,
    },
    {
        "slug": "market_vs_control",
        "title": "Market vs Control",
        "baseline_story": 0,
        "treated_story": 1,
    },
]

SPECIFICATIONS = [
    {
        "slug": "cats_only",
        "title": "Categories Only",
    },
    {
        "slug": "cats_hp",
        "title": "Categories + Beliefs HP",
    },
]

FULL_STORIES = [0, 1, 2, 4]


def comparison_stories(comparison: dict) -> list[int]:
    return [comparison["baseline_story"], comparison["treated_story"]]


def comparison_sample_note(comparison: dict) -> str:
    if comparison["slug"] == "aid_vs_bonus":
        return "Sample restricted to Bonus and Aid."
    return "Sample restricted to Control and Market."


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = pd.read_excel(P1_PATH)
    p2 = pd.read_excel(P2_PATH)

    for frame in [p1, p2]:
        for column in [
            "share_sent",
            "beliefs",
            "beliefs_hp",
            "choice",
            "choice_hp",
            "share_sent_hp",
            "share_sent_p1",
            "story",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    p1["category"] = p1["category"].replace(
        {
            "Cooperation": "Mutual Benefit / Cooperation",
            "Payoff Maximization": "Self-interest",
        }
    )
    p2["category"] = p2["category"].replace(
        {
            "Cooperation": "Mutual Benefit / Cooperation",
            "Payoff Maximization": "Self-interest",
        }
    )
    return p1, p2


def cleaned_subset(frame: pd.DataFrame, mask: pd.Series, needed: list[str]) -> pd.DataFrame:
    return frame.loc[mask].dropna(subset=needed).copy()


def mean_difference(treated: pd.Series, baseline: pd.Series) -> float:
    return float(treated.mean() - baseline.mean())


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 30:
        return np.nan, np.nan
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_limits(values: pd.DataFrame) -> tuple[float, float]:
    finite = values[["actual", "predicted"]].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1e-10:
        pad = 1.0 if abs(lo) < 1e-10 else abs(lo) * 0.15
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def significance_stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def format_estimate(model, term: str) -> tuple[str, str]:
    coef = model.params[term]
    se = model.bse[term]
    stars = significance_stars(model.pvalues[term])
    return f"{coef:.3f}{stars}", f"({se:.3f})"


def coef_with_stars(model, term: str) -> tuple[str, str]:
    if term not in model.params.index:
        return "", ""
    p = float(model.pvalues[term])
    if p < 0.01:
        stars = "$^{***}$"
    elif p < 0.05:
        stars = "$^{**}$"
    elif p < 0.10:
        stars = "$^{*}$"
    else:
        stars = ""
    return f"{model.params[term]:.3f}{stars}", f"({model.bse[term]:.3f})"


def p1_action_formula(spec_slug: str, game: str) -> tuple[str, list[str]]:
    if spec_slug == "cats_only" or game == "dgkw":
        return (
            "share_sent ~ C(category, Treatment(reference='Moral'))",
            ["share_sent", "category"],
        )
    return (
        "share_sent ~ C(category, Treatment(reference='Moral')) + beliefs_hp",
        ["share_sent", "category", "beliefs_hp"],
    )


def p1_belief_formula(spec_slug: str) -> tuple[str, list[str]]:
    if spec_slug == "cats_only":
        return (
            "beliefs ~ C(category, Treatment(reference='Moral'))",
            ["beliefs", "category"],
        )
    return (
        "beliefs ~ C(category, Treatment(reference='Moral')) + beliefs_hp",
        ["beliefs", "category", "beliefs_hp"],
    )


def p2_formula(spec_slug: str, game: str) -> tuple[str, list[str], str]:
    if game == "ug":
        if spec_slug == "cats_only":
            return (
                "choice ~ C(category, Treatment(reference='Moral good'))",
                ["choice", "category"],
                "choice",
            )
        return (
            "choice ~ C(category, Treatment(reference='Moral good')) + choice_hp",
            ["choice", "category", "choice_hp"],
            "choice",
        )
    if spec_slug == "cats_only":
        return (
            "share_sent ~ C(category, Treatment(reference='Moral good'))",
            ["share_sent", "category"],
            "share_sent",
        )
    return (
        "share_sent ~ C(category, Treatment(reference='Moral good')) + share_sent_hp",
        ["share_sent", "category", "share_sent_hp"],
        "share_sent",
    )


def build_sender_receiver_merge(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    sender = p1[
        ["PROLIFIC_PID", "story", "game", "category", "beliefs", "beliefs_hp"]
    ].copy()
    receiver = p2[
        ["P1_ID", "story", "game", "choice", "choice_hp", "share_sent", "share_sent_hp"]
    ].copy()
    merged = sender.merge(
        receiver,
        left_on=["PROLIFIC_PID", "story", "game"],
        right_on=["P1_ID", "story", "game"],
        how="inner",
        suffixes=("", "_p2"),
    )
    merged["forecast_error_hyp"] = np.where(
        merged["game"].eq("ug"),
        merged["beliefs_hp"] - merged["choice_hp"],
        merged["beliefs_hp"] - merged["share_sent_hp"],
    )
    merged["forecast_error"] = np.where(
        merged["game"].eq("ug"),
        merged["beliefs"] - merged["choice"],
        merged["beliefs"] - merged["share_sent"],
    )
    return merged


def forecast_error_formula(spec_slug: str, outcome: str) -> tuple[str, list[str]]:
    if spec_slug == "cats_only":
        return (
            f"{outcome} ~ C(category, Treatment(reference='Moral'))",
            [outcome, "category"],
        )
    return (
        f"{outcome} ~ C(category, Treatment(reference='Moral')) + beliefs_hp",
        [outcome, "category", "beliefs_hp"],
    )


def fit_effect(
    fit_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    treated_frame: pd.DataFrame,
    formula: str,
    outcome: str,
    rng: np.random.Generator,
    actual_baseline_frame: pd.DataFrame | None = None,
    actual_treated_frame: pd.DataFrame | None = None,
) -> dict:
    if actual_baseline_frame is None:
        actual_baseline_frame = baseline_frame
    if actual_treated_frame is None:
        actual_treated_frame = treated_frame

    model = smf.ols(formula, data=fit_frame).fit()
    pred_baseline = model.predict(baseline_frame)
    pred_treated = model.predict(treated_frame)
    actual = mean_difference(actual_treated_frame[outcome], actual_baseline_frame[outcome])
    predicted = float(pred_treated.mean() - pred_baseline.mean())

    actual_boot = []
    predicted_boot = []
    fit_idx = np.arange(len(fit_frame))
    baseline_idx = np.arange(len(baseline_frame))
    treated_idx = np.arange(len(treated_frame))
    actual_baseline_idx = np.arange(len(actual_baseline_frame))
    actual_treated_idx = np.arange(len(actual_treated_frame))

    for _ in range(BOOTSTRAP_REPS):
        fit_boot = fit_frame.iloc[rng.choice(fit_idx, size=len(fit_idx), replace=True)].copy()
        baseline_boot = baseline_frame.iloc[rng.choice(baseline_idx, size=len(baseline_idx), replace=True)].copy()
        treated_boot = treated_frame.iloc[rng.choice(treated_idx, size=len(treated_idx), replace=True)].copy()
        actual_baseline_boot = actual_baseline_frame.iloc[
            rng.choice(actual_baseline_idx, size=len(actual_baseline_idx), replace=True)
        ].copy()
        actual_treated_boot = actual_treated_frame.iloc[
            rng.choice(actual_treated_idx, size=len(actual_treated_idx), replace=True)
        ].copy()
        try:
            boot_model = smf.ols(formula, data=fit_boot).fit()
            boot_pred_baseline = boot_model.predict(baseline_boot)
            boot_pred_treated = boot_model.predict(treated_boot)
            predicted_boot.append(float(boot_pred_treated.mean() - boot_pred_baseline.mean()))
            actual_boot.append(mean_difference(actual_treated_boot[outcome], actual_baseline_boot[outcome]))
        except Exception:
            continue

    actual_ci_low, actual_ci_high = bootstrap_ci(actual_boot)
    predicted_ci_low, predicted_ci_high = bootstrap_ci(predicted_boot)

    return {
        "actual": actual,
        "predicted": predicted,
        "gap": float(predicted - actual),
        "actual_ci_low": actual_ci_low,
        "actual_ci_high": actual_ci_high,
        "predicted_ci_low": predicted_ci_low,
        "predicted_ci_high": predicted_ci_high,
        "model": model,
    }


def fit_p1_action_models_for_mapping(
    p1: pd.DataFrame,
    spec_slug: str,
    comparison: dict,
) -> dict[str, object]:
    fit_mask = p1["story"].isin(comparison_stories(comparison))
    models: dict[str, object] = {}
    for game in ["ug", "tg"]:
        formula, needed = p1_action_formula(spec_slug, game)
        fit = cleaned_subset(p1, p1["game"].eq(game) & fit_mask, needed)
        models[game] = smf.ols(formula, data=fit).fit()
    return models


def attach_predicted_sender_offer(
    p2: pd.DataFrame,
    p1: pd.DataFrame,
    spec_slug: str,
    comparison: dict,
) -> pd.DataFrame:
    models = fit_p1_action_models_for_mapping(p1, spec_slug, comparison)
    fit_mask = p1["story"].isin(comparison_stories(comparison))
    out = p2.copy()

    for game in ["ug", "tg"]:
        formula, needed = p1_action_formula(spec_slug, game)
        fit = cleaned_subset(p1, p1["game"].eq(game) & fit_mask, needed).copy()
        fit["p1_fitted"] = models[game].predict(fit)
        mapping = (
            fit.groupby("share_sent", as_index=False)["p1_fitted"]
            .mean()
            .rename(columns={"share_sent": "share_sent_p1", "p1_fitted": "predicted_share_sent_p1"})
        )
        mask = out["game"].eq(game)
        rounded_offer = pd.to_numeric(out.loc[mask, "share_sent_p1"], errors="coerce").round(10)
        lookup = mapping.assign(share_sent_p1_round=mapping["share_sent_p1"].round(10)).set_index("share_sent_p1_round")[
            "predicted_share_sent_p1"
        ]
        out.loc[mask, "predicted_share_sent_p1"] = rounded_offer.map(lookup).to_numpy()

    return out


def compute_p1_results(
    p1: pd.DataFrame,
    comparison: dict,
    spec_slug: str,
) -> tuple[pd.DataFrame, list[tuple[str, object]]]:
    rows = []
    models = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + comparison["treated_story"] + (0 if spec_slug == "cats_only" else 50))
    fit_mask = p1["story"].isin(comparison_stories(comparison))

    for game, label, display in [("dgkw", "KW action", "KW"), ("ug", "UG action", "UG"), ("tg", "TG action", "TG")]:
        formula, needed = p1_action_formula(spec_slug, game)
        fit = cleaned_subset(p1, p1["game"].eq(game) & fit_mask, needed)
        baseline = cleaned_subset(p1, p1["game"].eq(game) & p1["story"].eq(comparison["baseline_story"]), needed)
        treated = cleaned_subset(p1, p1["game"].eq(game) & p1["story"].eq(comparison["treated_story"]), needed)
        actual_baseline = cleaned_subset(
            p1,
            p1["game"].eq(game) & p1["story"].eq(comparison["baseline_story"]),
            ["share_sent"],
        )
        actual_treated = cleaned_subset(
            p1,
            p1["game"].eq(game) & p1["story"].eq(comparison["treated_story"]),
            ["share_sent"],
        )
        stats = fit_effect(
            fit,
            baseline,
            treated,
            formula,
            "share_sent",
            rng,
            actual_baseline_frame=actual_baseline,
            actual_treated_frame=actual_treated,
        )
        rows.append(
            {
                "panel": "action",
                "game": display,
                "game_code": game,
                "label": f"{display} share sent",
                "actual": stats["actual"],
                "predicted": stats["predicted"],
                "gap": stats["gap"],
                "actual_ci_low": stats["actual_ci_low"],
                "actual_ci_high": stats["actual_ci_high"],
                "predicted_ci_low": stats["predicted_ci_low"],
                "predicted_ci_high": stats["predicted_ci_high"],
            }
        )
        models.append((label, stats["model"]))

    formula, needed = p1_belief_formula(spec_slug)
    for game, label, display in [("ug", "UG beliefs", "UG"), ("tg", "TG beliefs", "TG")]:
        fit = cleaned_subset(p1, p1["game"].eq(game) & fit_mask, needed)
        baseline = cleaned_subset(p1, p1["game"].eq(game) & p1["story"].eq(comparison["baseline_story"]), needed)
        treated = cleaned_subset(p1, p1["game"].eq(game) & p1["story"].eq(comparison["treated_story"]), needed)
        actual_baseline = cleaned_subset(
            p1,
            p1["game"].eq(game) & p1["story"].eq(comparison["baseline_story"]),
            ["beliefs"],
        )
        actual_treated = cleaned_subset(
            p1,
            p1["game"].eq(game) & p1["story"].eq(comparison["treated_story"]),
            ["beliefs"],
        )
        stats = fit_effect(
            fit,
            baseline,
            treated,
            formula,
            "beliefs",
            rng,
            actual_baseline_frame=actual_baseline,
            actual_treated_frame=actual_treated,
        )
        rows.append(
            {
                "panel": "belief",
                "game": display,
                "game_code": game,
                "label": f"{display} beliefs",
                "actual": stats["actual"],
                "predicted": stats["predicted"],
                "gap": stats["gap"],
                "actual_ci_low": stats["actual_ci_low"],
                "actual_ci_high": stats["actual_ci_high"],
                "predicted_ci_low": stats["predicted_ci_low"],
                "predicted_ci_high": stats["predicted_ci_high"],
            }
        )
        models.append((label, stats["model"]))

    return pd.DataFrame(rows), models


def compute_p2_results(
    p2: pd.DataFrame,
    comparison: dict,
    spec_slug: str,
) -> tuple[pd.DataFrame, list[tuple[str, object]]]:
    rows = []
    models = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + comparison["treated_story"] + (0 if spec_slug == "cats_only" else 50))
    fit_mask = p2["story"].isin(comparison_stories(comparison))

    for game, label, display in [("ug", "UG outcome", "UG"), ("tg", "TG outcome", "TG")]:
        formula, needed, outcome = p2_formula(spec_slug, game)
        fit = cleaned_subset(p2, p2["game"].eq(game) & fit_mask, needed)
        baseline = cleaned_subset(p2, p2["game"].eq(game) & p2["story"].eq(comparison["baseline_story"]), needed)
        treated = cleaned_subset(p2, p2["game"].eq(game) & p2["story"].eq(comparison["treated_story"]), needed)
        actual_baseline = cleaned_subset(
            p2,
            p2["game"].eq(game) & p2["story"].eq(comparison["baseline_story"]),
            [outcome],
        )
        actual_treated = cleaned_subset(
            p2,
            p2["game"].eq(game) & p2["story"].eq(comparison["treated_story"]),
            [outcome],
        )
        stats = fit_effect(
            fit,
            baseline,
            treated,
            formula,
            outcome,
            rng,
            actual_baseline_frame=actual_baseline,
            actual_treated_frame=actual_treated,
        )
        rows.append(
            {
                "game": display,
                "game_code": game,
                "label": "UG acceptance" if game == "ug" else "TG share sent",
                "actual": stats["actual"],
                "predicted": stats["predicted"],
                "gap": stats["gap"],
                "actual_ci_low": stats["actual_ci_low"],
                "actual_ci_high": stats["actual_ci_high"],
                "predicted_ci_low": stats["predicted_ci_low"],
                "predicted_ci_high": stats["predicted_ci_high"],
            }
        )
        models.append((label, stats["model"]))

    return pd.DataFrame(rows), models


def p2_pred_sender_formula(spec_slug: str, game: str) -> tuple[str, list[str], str]:
    if game == "ug":
        if spec_slug == "cats_only":
            return (
                "choice ~ C(category, Treatment(reference='Moral good')) + predicted_share_sent_p1",
                ["choice", "category", "predicted_share_sent_p1"],
                "choice",
            )
        return (
            "choice ~ C(category, Treatment(reference='Moral good')) + choice_hp + predicted_share_sent_p1",
            ["choice", "category", "choice_hp", "predicted_share_sent_p1"],
            "choice",
        )
    if spec_slug == "cats_only":
        return (
            "share_sent ~ C(category, Treatment(reference='Moral good')) + predicted_share_sent_p1",
            ["share_sent", "category", "predicted_share_sent_p1"],
            "share_sent",
        )
    return (
        "share_sent ~ C(category, Treatment(reference='Moral good')) + share_sent_hp + predicted_share_sent_p1",
        ["share_sent", "category", "share_sent_hp", "predicted_share_sent_p1"],
        "share_sent",
    )


def compute_p2_predsender_results(
    p2: pd.DataFrame,
    p1: pd.DataFrame,
    comparison: dict,
    spec_slug: str,
) -> tuple[pd.DataFrame, list[tuple[str, object]]]:
    rows = []
    models = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200 + comparison["treated_story"] + (0 if spec_slug == "cats_only" else 50))
    fit_mask = p2["story"].isin(comparison_stories(comparison))
    augmented = attach_predicted_sender_offer(p2, p1, spec_slug, comparison)

    for game, label, display in [("ug", "UG outcome", "UG"), ("tg", "TG outcome", "TG")]:
        formula, needed, outcome = p2_pred_sender_formula(spec_slug, game)
        fit = cleaned_subset(augmented, augmented["game"].eq(game) & fit_mask, needed)
        baseline = cleaned_subset(augmented, augmented["game"].eq(game) & augmented["story"].eq(comparison["baseline_story"]), needed)
        treated = cleaned_subset(augmented, augmented["game"].eq(game) & augmented["story"].eq(comparison["treated_story"]), needed)
        actual_baseline = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["baseline_story"]),
            [outcome],
        )
        actual_treated = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["treated_story"]),
            [outcome],
        )
        stats = fit_effect(
            fit,
            baseline,
            treated,
            formula,
            outcome,
            rng,
            actual_baseline_frame=actual_baseline,
            actual_treated_frame=actual_treated,
        )
        rows.append(
            {
                "game": display,
                "game_code": game,
                "label": "UG acceptance" if game == "ug" else "TG share sent",
                "actual": stats["actual"],
                "predicted": stats["predicted"],
                "gap": stats["gap"],
                "actual_ci_low": stats["actual_ci_low"],
                "actual_ci_high": stats["actual_ci_high"],
                "predicted_ci_low": stats["predicted_ci_low"],
                "predicted_ci_high": stats["predicted_ci_high"],
            }
        )
        models.append((label, stats["model"]))

    return pd.DataFrame(rows), models


def p2_hyp_predsender_formula(game: str) -> tuple[str, list[str], str]:
    if game == "ug":
        return (
            "choice_hp ~ C(category, Treatment(reference='Moral good')) + predicted_share_sent_p1",
            ["choice_hp", "category", "predicted_share_sent_p1"],
            "choice_hp",
        )
    return (
        "share_sent_hp ~ C(category, Treatment(reference='Moral good')) + predicted_share_sent_p1",
        ["share_sent_hp", "category", "predicted_share_sent_p1"],
        "share_sent_hp",
    )


def compute_p2_hyp_predsender_results(
    p2: pd.DataFrame,
    p1: pd.DataFrame,
    comparison: dict,
    spec_slug: str,
) -> tuple[pd.DataFrame, list[tuple[str, object]]]:
    rows = []
    models = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 300 + comparison["treated_story"] + (0 if spec_slug == "cats_only" else 50))
    fit_mask = p2["story"].isin(comparison_stories(comparison))
    augmented = attach_predicted_sender_offer(p2, p1, spec_slug, comparison)

    for game, label, display in [("ug", "UG hypothetical action", "UG"), ("tg", "TG hypothetical action", "TG")]:
        formula, needed, outcome = p2_hyp_predsender_formula(game)
        fit = cleaned_subset(augmented, augmented["game"].eq(game) & fit_mask, needed)
        baseline = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["baseline_story"]),
            needed,
        )
        treated = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["treated_story"]),
            needed,
        )
        actual_baseline = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["baseline_story"]),
            [outcome],
        )
        actual_treated = cleaned_subset(
            augmented,
            augmented["game"].eq(game) & augmented["story"].eq(comparison["treated_story"]),
            [outcome],
        )
        stats = fit_effect(
            fit,
            baseline,
            treated,
            formula,
            outcome,
            rng,
            actual_baseline_frame=actual_baseline,
            actual_treated_frame=actual_treated,
        )
        rows.append(
            {
                "game": display,
                "game_code": game,
                "label": "UG hypothetical action" if game == "ug" else "TG hypothetical action",
                "actual": stats["actual"],
                "predicted": stats["predicted"],
                "gap": stats["gap"],
                "actual_ci_low": stats["actual_ci_low"],
                "actual_ci_high": stats["actual_ci_high"],
                "predicted_ci_low": stats["predicted_ci_low"],
                "predicted_ci_high": stats["predicted_ci_high"],
            }
        )
        models.append((label, stats["model"]))

    return pd.DataFrame(rows), models


def make_p1_figure(rows: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    fig.subplots_adjust(wspace=0.24, bottom=0.18)

    for ax, panel, title in [
        (axes[0], "action", "Player 1 action"),
        (axes[1], "belief", "Player 1 beliefs"),
    ]:
        subset = rows[rows["panel"] == panel].copy()
        x = np.arange(len(subset))
        width = 0.34
        actual = subset["actual"].to_numpy(dtype=float) * 100
        predicted = subset["predicted"].to_numpy(dtype=float) * 100
        actual_lower = actual - subset["actual_ci_low"].to_numpy(dtype=float) * 100
        actual_upper = subset["actual_ci_high"].to_numpy(dtype=float) * 100 - actual
        predicted_lower = predicted - subset["predicted_ci_low"].to_numpy(dtype=float) * 100
        predicted_upper = subset["predicted_ci_high"].to_numpy(dtype=float) * 100 - predicted
        actual_colors = [GAME_COLORS[code] for code in subset["game_code"]]
        predicted_colors = [GAME_COLORS_BRIGHT[code] for code in subset["game_code"]]

        ax.bar(x - width / 2, actual, width=width, color=actual_colors, edgecolor="black", linewidth=0.7)
        ax.bar(x + width / 2, predicted, width=width, color=predicted_colors, edgecolor="black", linewidth=0.7, hatch="///")
        ax.errorbar(x - width / 2, actual, yerr=[actual_lower, actual_upper], fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3, zorder=4)
        ax.errorbar(x + width / 2, predicted, yerr=[predicted_lower, predicted_upper], fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3, zorder=4)
        ax.axhline(0, color="#666666", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["game"].tolist())
        ax.set_ylabel("Percentage points")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor="#666666", edgecolor="black", label="Actual"),
        Patch(facecolor="#D0D0D0", edgecolor="black", hatch="///", label="Predicted"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_p2_figure(rows: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    fig.subplots_adjust(bottom=0.18)
    x = np.arange(len(rows))
    width = 0.34
    actual = rows["actual"].to_numpy(dtype=float) * 100
    predicted = rows["predicted"].to_numpy(dtype=float) * 100
    actual_lower = actual - rows["actual_ci_low"].to_numpy(dtype=float) * 100
    actual_upper = rows["actual_ci_high"].to_numpy(dtype=float) * 100 - actual
    predicted_lower = predicted - rows["predicted_ci_low"].to_numpy(dtype=float) * 100
    predicted_upper = rows["predicted_ci_high"].to_numpy(dtype=float) * 100 - predicted
    actual_colors = [GAME_COLORS[code] for code in rows["game_code"]]
    predicted_colors = [GAME_COLORS_BRIGHT[code] for code in rows["game_code"]]

    ax.bar(x - width / 2, actual, width=width, color=actual_colors, edgecolor="black", linewidth=0.7)
    ax.bar(x + width / 2, predicted, width=width, color=predicted_colors, edgecolor="black", linewidth=0.7, hatch="///")
    ax.errorbar(x - width / 2, actual, yerr=[actual_lower, actual_upper], fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3, zorder=4)
    ax.errorbar(x + width / 2, predicted, yerr=[predicted_lower, predicted_upper], fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3, zorder=4)
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["game"].tolist())
    ax.set_ylabel("Percentage points")
    ax.set_title("Player 2 outcomes")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(facecolor="#666666", edgecolor="black", label="Actual"),
        Patch(facecolor="#D0D0D0", edgecolor="black", hatch="///", label="Predicted"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_instability_figure(
    rows: pd.DataFrame,
    player: str,
    spec: dict,
    output_path: Path,
    title_override: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    subset = rows.copy()
    lo, hi = compute_limits(subset)
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1)

    for comparison in COMPARISONS:
        comp_rows = subset[subset["comparison_slug"] == comparison["slug"]]
        ax.scatter(
            comp_rows["actual"],
            comp_rows["predicted"],
            s=52,
            color=COMPARISON_COLORS[comparison["slug"]],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label=comparison["title"],
        )
        for _, row in comp_rows.iterrows():
            ax.annotate(
                row["point_label"],
                (row["actual"], row["predicted"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9 * FONT_SCALE,
                color=GAME_COLORS[row["game_code"]],
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual treatment effect")
    ax.set_ylabel("Predicted treatment effect")
    ax.set_title(title_override or f"{player}, {spec['title']}")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_instability_panel_figure(
    panels: list[tuple[pd.DataFrame, str]],
    output_path: Path,
) -> None:
    # 2x2 combined version of make_instability_figure (whose single-panel calls are
    # kept, commented, in main); drawn ~13.6in wide and included at \textwidth, so
    # each panel shows at ~3.2in against the singles' ~4.8in --- the font boost
    # restores the same apparent text size.
    boost = 1.45
    with plt.rc_context(
        {
            "axes.titlesize": 12 * FONT_SCALE * boost,
            "axes.labelsize": 10 * FONT_SCALE * boost,
            "xtick.labelsize": 9 * FONT_SCALE * boost,
            "ytick.labelsize": 9 * FONT_SCALE * boost,
            "legend.fontsize": 9 * FONT_SCALE * boost,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.6))
        for ax, (rows, panel_title) in zip(axes.flat, panels):
            subset = rows.copy()
            lo, hi = compute_limits(subset)
            ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1)
            for comparison in COMPARISONS:
                comp_rows = subset[subset["comparison_slug"] == comparison["slug"]]
                ax.scatter(
                    comp_rows["actual"],
                    comp_rows["predicted"],
                    s=52,
                    color=COMPARISON_COLORS[comparison["slug"]],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                    label=comparison["title"],
                )
                for _, row in comp_rows.iterrows():
                    ax.annotate(
                        row["point_label"],
                        (row["actual"], row["predicted"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=9 * FONT_SCALE * boost,
                        color=GAME_COLORS[row["game_code"]],
                    )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("Actual treatment effect")
            ax.set_ylabel("Predicted treatment effect")
            ax.set_title(panel_title)
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def write_instability_regression_table(rows: pd.DataFrame, spec: dict, output_path: Path) -> None:
    subsets = {
        "All": rows.copy(),
        "Player 1": rows[rows["player"] == "P1"].copy(),
        "Player 2": rows[rows["player"] == "P2"].copy(),
    }
    models = {label: smf.ols("actual ~ predicted", data=data).fit() for label, data in subsets.items()}
    formatted = {}
    for label, model in models.items():
        formatted[label] = {
            "intercept": format_estimate(model, "Intercept"),
            "predicted": format_estimate(model, "predicted"),
            "n": int(model.nobs),
            "r2": model.rsquared,
        }

    table = rf"""\begin{{table}}[!htbp]
\centering
\caption{{\textbf{{Instability calibration regressions, {spec['title']}}}}}
\label{{tab:{spec['slug']}_instability_regs}}
\begin{{tabular}}{{lccc}}
\toprule
& All & Player 1 & Player 2 \\
\midrule
Intercept & {formatted["All"]["intercept"][0]} & {formatted["Player 1"]["intercept"][0]} & {formatted["Player 2"]["intercept"][0]} \\
& {formatted["All"]["intercept"][1]} & {formatted["Player 1"]["intercept"][1]} & {formatted["Player 2"]["intercept"][1]} \\
Predicted & {formatted["All"]["predicted"][0]} & {formatted["Player 1"]["predicted"][0]} & {formatted["Player 2"]["predicted"][0]} \\
& {formatted["All"]["predicted"][1]} & {formatted["Player 1"]["predicted"][1]} & {formatted["Player 2"]["predicted"][1]} \\
Observations & {formatted["All"]["n"]} & {formatted["Player 1"]["n"]} & {formatted["Player 2"]["n"]} \\
$R^2$ & {formatted["All"]["r2"]:.3f} & {formatted["Player 1"]["r2"]:.3f} & {formatted["Player 2"]["r2"]:.3f} \\
\bottomrule
\end{{tabular}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.9\textwidth}}
\footnotesize Notes: Unit of observation is an outcome-by-comparison cell from the fitted-values exercise. Each column reports the regression of the actual treatment effect on the predicted treatment effect. Standard errors in parentheses. $^*$ $p<0.1$, $^{{**}}$ $p<0.05$, $^{{***}}$ $p<0.01$.
\end{{minipage}}
\end{{table}}
"""
    output_path.write_text(table, encoding="utf-8")


def write_fit_table(rows: pd.DataFrame, caption: str, label: str, output_path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{{caption}}}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Outcome & Actual & Predicted & Gap & Predicted / Actual \\\\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        ratio = np.nan if abs(row["actual"]) < 1e-12 else row["predicted"] / row["actual"]
        ratio_text = "" if pd.isna(ratio) else f"{ratio:.3f}"
        lines.append(
            f"{row['label']} & {row['actual']:.3f} & {row['predicted']:.3f} & {row['gap']:.3f} & {ratio_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_note(output_path: Path, spec: dict) -> None:
    if spec["slug"] == "cats_only":
        text = (
            "Sample used for estimation: pooled Control, Market, Bonus, and Aid. "
            "Player 1 action equations are estimated by game using category dummies only. "
            "Player 1 belief equations are estimated by game using category dummies only. "
            "Player 2 outcome equations are estimated by game using category dummies only."
        )
    else:
        text = (
            "Sample used for estimation: pooled Control, Market, Bonus, and Aid. "
            "Player 1 action equations are estimated by game. In KW, share sent is regressed on categories only. "
            "In UG and TG, share sent is regressed on categories and hypothetical beliefs. "
            "Player 1 belief equations are estimated by game using categories and hypothetical beliefs. "
            "Player 2 outcome equations are estimated by game using categories and the corresponding hypothetical response variable."
        )
    output_path.write_text("\n".join(["\\begin{flushleft}", f"\\footnotesize {text}", "\\end{flushleft}", ""]), encoding="utf-8")


def write_p1_reg_table(models: list[tuple[str, object]], comparison: dict, spec: dict, output_path: Path) -> None:
    rows = [
        ("Self-interest", "C(category, Treatment(reference='Moral'))[T.Self-interest]"),
        ("Mutual Benefit / Cooperation", "C(category, Treatment(reference='Moral'))[T.Mutual Benefit / Cooperation]"),
        ("No clear justification", "C(category, Treatment(reference='Moral'))[T.No clear justification]"),
        ("Hypothetical beliefs", "beliefs_hp"),
        ("Constant", "Intercept"),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{Benchmark Regressions: Player 1, {comparison['title']}, {spec['title']}}}}}",
        f"\\label{{tab:{spec['slug']}_{comparison['slug']}_fullpooled_p1_regs}}",
        "\\resizebox{0.98\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & (1) & (2) & (3) & (4) & (5) \\\\",
        " & KW action & UG action & TG action & UG beliefs & TG beliefs \\\\",
        "\\midrule",
    ]
    for row_label, term in rows:
        coef_row = [row_label]
        se_row = [""]
        for _, model in models:
            coef, se = coef_with_stars(model, term)
            coef_row.append(coef)
            se_row.append(se)
        if any(cell != "" for cell in coef_row[1:]):
            lines.append(" & ".join(coef_row) + " \\\\")
            lines.append(" & ".join(se_row) + " \\\\")
    lines.extend(
        [
            "\\midrule",
            "Observations & " + " & ".join(str(int(model.nobs)) for _, model in models) + " \\\\",
            "$R^2$ & " + " & ".join(f"{model.rsquared:.3f}" for _, model in models) + " \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\begin{flushleft}",
            (
                f"\\footnotesize Standard errors in parentheses. Baseline category is Moral. {comparison_sample_note(comparison)} All equations use only category dummies. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$."
                if spec["slug"] == "cats_only"
                else f"\\footnotesize Standard errors in parentheses. Baseline category is Moral. {comparison_sample_note(comparison)} UG and TG equations include hypothetical beliefs in addition to category dummies; KW action uses category dummies only. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$."
            ),
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_p2_reg_table(
    models: list[tuple[str, object]],
    comparison: dict,
    spec: dict,
    output_path: Path,
    augmented_models: list[tuple[str, object]] | None = None,
) -> None:
    rows = [
        ("Moral bad", "C(category, Treatment(reference='Moral good'))[T.Moral bad]"),
        ("Mutual Benefit / Cooperation", "C(category, Treatment(reference='Moral good'))[T.Mutual Benefit / Cooperation]"),
        ("No clear justification", "C(category, Treatment(reference='Moral good'))[T.No clear justification]"),
        ("Self-interest", "C(category, Treatment(reference='Moral good'))[T.Self-interest]"),
        ("Hypothetical acceptance", "choice_hp"),
        ("Hypothetical share sent P2", "share_sent_hp"),
        ("Predicted share sent P1", "predicted_share_sent_p1"),
        ("Constant", "Intercept"),
    ]
    all_models = models if augmented_models is None else [*models, *augmented_models]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{Benchmark Regressions: Player 2, {comparison['title']}, {spec['title']}}}}}",
        f"\\label{{tab:{spec['slug']}_{comparison['slug']}_fullpooled_p2_regs}}",
        "\\resizebox{0.94\\textwidth}{!}{%",
        "\\begin{tabular}{l" + ("c" * len(all_models)) + "}",
        "\\toprule",
        " & " + " & ".join(f"({idx})" for idx in range(1, len(all_models) + 1)) + " \\\\",
        (
            " & UG outcome & TG outcome"
            if augmented_models is None
            else " & UG outcome & TG outcome & UG outcome + fitted share sent P1 & TG outcome + fitted share sent P1"
        )
        + " \\\\",
        "\\midrule",
    ]
    for row_label, term in rows:
        coef_row = [row_label]
        se_row = [""]
        for _, model in all_models:
            coef, se = coef_with_stars(model, term)
            coef_row.append(coef)
            se_row.append(se)
        if any(cell != "" for cell in coef_row[1:]):
            lines.append(" & ".join(coef_row) + " \\\\")
            lines.append(" & ".join(se_row) + " \\\\")
    if augmented_models is None:
        note_body = (
            f"\\footnotesize Standard errors in parentheses. Baseline category is Moral good. {comparison_sample_note(comparison)} All equations use only category dummies. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$."
            if spec["slug"] == "cats_only"
            else f"\\footnotesize Standard errors in parentheses. Baseline category is Moral good. {comparison_sample_note(comparison)} UG equations include hypothetical acceptance and TG equations include hypothetical share sent P2, in addition to category dummies. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$."
        )
    else:
        note_body = (
            f"\\footnotesize Standard errors in parentheses. Baseline category is Moral good. {comparison_sample_note(comparison)} "
            "Columns (3) and (4) add fitted share sent P1, computed from the comparison-specific Player 1 action regression in the same specification by averaging fitted values within each observed Player 1 donation level. "
            + (
                "Columns (1) and (2) use only category dummies, while columns (3) and (4) add fitted share sent P1. "
                if spec["slug"] == "cats_only"
                else "UG equations include hypothetical acceptance and TG equations include hypothetical share sent P2; columns (3) and (4) additionally include fitted share sent P1. "
            )
            + "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."
        )
    lines.extend(
        [
            "\\midrule",
            "Observations & " + " & ".join(str(int(model.nobs)) for _, model in all_models) + " \\\\",
            "$R^2$ & " + " & ".join(f"{model.rsquared:.3f}" for _, model in all_models) + " \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\begin{flushleft}",
            note_body,
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_p2_predsender_reg_table(models: list[tuple[str, object]], comparison: dict, spec: dict, output_path: Path) -> None:
    rows = [
        ("Moral bad", "C(category, Treatment(reference='Moral good'))[T.Moral bad]"),
        ("Mutual Benefit / Cooperation", "C(category, Treatment(reference='Moral good'))[T.Mutual Benefit / Cooperation]"),
        ("No clear justification", "C(category, Treatment(reference='Moral good'))[T.No clear justification]"),
        ("Self-interest", "C(category, Treatment(reference='Moral good'))[T.Self-interest]"),
        ("Hypothetical acceptance", "choice_hp"),
        ("Hypothetical share sent P2", "share_sent_hp"),
        ("Predicted share sent P1", "predicted_share_sent_p1"),
        ("Constant", "Intercept"),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{Player 2 Regressions with Predicted Player 1 Allocation, {comparison['title']}, {spec['title']}}}}}",
        f"\\label{{tab:{spec['slug']}_{comparison['slug']}_fullpooled_p2_predsender_regs}}",
        "\\resizebox{0.84\\textwidth}{!}{%",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & (1) & (2) \\\\",
        " & UG outcome & TG outcome \\\\",
        "\\midrule",
    ]
    for row_label, term in rows:
        coef_row = [row_label]
        se_row = [""]
        for _, model in models:
            coef, se = coef_with_stars(model, term)
            coef_row.append(coef)
            se_row.append(se)
        if any(cell != "" for cell in coef_row[1:]):
            lines.append(" & ".join(coef_row) + " \\\\")
            lines.append(" & ".join(se_row) + " \\\\")
    note = (
        f"\\footnotesize Standard errors in parentheses. Baseline category is Moral good. {comparison_sample_note(comparison)} "
        "Predicted share sent P1 is computed from the comparison-specific Player 1 action regression in the same specification, averaging fitted values within each observed Player 1 donation level. "
        + (
            "All equations also include category dummies only. "
            if spec["slug"] == "cats_only"
            else "UG equations also include hypothetical acceptance and TG equations also include hypothetical share sent P2. "
        )
        + "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."
    )
    lines.extend(
        [
            "\\midrule",
            "Observations & " + " & ".join(str(int(model.nobs)) for _, model in models) + " \\\\",
            "$R^2$ & " + " & ".join(f"{model.rsquared:.3f}" for _, model in models) + " \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\begin{flushleft}",
            note,
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def compute_forecast_error_models(
    merged: pd.DataFrame,
    spec_slug: str,
    outcome: str,
) -> list[tuple[str, object]]:
    models = []
    for game, label in [("ug", "UG"), ("tg", "TG")]:
        formula, needed = forecast_error_formula(spec_slug, outcome)
        fit = cleaned_subset(merged, merged["game"].eq(game) & merged["story"].isin(FULL_STORIES), needed)
        model = smf.ols(formula, data=fit).fit()
        models.append((label, model))
    return models


def write_forecast_error_reg_table(
    models: list[tuple[str, object]],
    spec: dict,
    title: str,
    label_stub: str,
    output_path: Path,
) -> None:
    rows = [
        ("Self-interest", "C(category, Treatment(reference='Moral'))[T.Self-interest]"),
        ("Mutual Benefit / Cooperation", "C(category, Treatment(reference='Moral'))[T.Mutual Benefit / Cooperation]"),
        ("No clear justification", "C(category, Treatment(reference='Moral'))[T.No clear justification]"),
        ("Hypothetical beliefs", "beliefs_hp"),
        ("Constant", "Intercept"),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{\\textbf{{Benchmark Regressions: Player 1, {title}, {spec['title']}}}}}",
        f"\\label{{tab:{spec['slug']}_{label_stub}}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & (1) & (2) \\\\",
        " & UG & TG \\\\",
        "\\midrule",
    ]
    for row_label, term in rows:
        coef_row = [row_label]
        se_row = [""]
        for _, model in models:
            coef, se = coef_with_stars(model, term)
            coef_row.append(coef)
            se_row.append(se)
        if any(cell != "" for cell in coef_row[1:]):
            lines.append(" & ".join(coef_row) + " \\\\")
            lines.append(" & ".join(se_row) + " \\\\")
    note_title = (
        "Hypothetical FE is defined as Player 1 hypothetical belief minus the corresponding hypothetical Player 2 response."
        if title == "Hypothetical FE"
        else "Forecast Error is defined as Player 1 belief minus realized Player 2 behavior."
    )
    lines.extend(
        [
            "\\midrule",
            "Observations & " + " & ".join(str(int(model.nobs)) for _, model in models) + " \\\\",
            "$R^2$ & " + " & ".join(f"{model.rsquared:.3f}" for _, model in models) + " \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{flushleft}",
            (
                f"\\footnotesize {note_title} Baseline category is Moral. Sample restricted to pooled Control, Market, Bonus, and Aid. "
                + (
                    "Regressions use only category dummies. "
                    if spec["slug"] == "cats_only"
                    else "Regressions include category dummies and hypothetical beliefs. "
                )
                + "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."
            ),
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    p1, p2 = load_data()
    merged = build_sender_receiver_merge(p1, p2)
    instability_points: dict[str, dict[str, list[pd.DataFrame]]] = {
        spec["slug"]: {
            "p1_action": [],
            "p1_belief": [],
            "p2_actual": [],
            "p2_hyp": [],
        }
        for spec in SPECIFICATIONS
    }

    for spec in SPECIFICATIONS:
        hyp_fe_models = compute_forecast_error_models(merged, spec["slug"], "forecast_error_hyp")
        fe_models = compute_forecast_error_models(merged, spec["slug"], "forecast_error")
        write_forecast_error_reg_table(
            hyp_fe_models,
            spec,
            "Hypothetical FE",
            f"{spec['slug']}_hypfe_regs",
            TEX_DIR / f"{spec['slug']}_hypfe_regs.tex",
        )
        write_forecast_error_reg_table(
            fe_models,
            spec,
            "Forecast Error",
            f"{spec['slug']}_fe_regs",
            TEX_DIR / f"{spec['slug']}_fe_regs.tex",
        )
        for comparison in COMPARISONS:
            p1_rows, p1_models = compute_p1_results(p1, comparison, spec["slug"])
            p2_rows, p2_models = compute_p2_results(p2, comparison, spec["slug"])
            p2_pred_rows, p2_pred_models = compute_p2_predsender_results(p2, p1, comparison, spec["slug"])
            p2_hyp_rows, _ = compute_p2_hyp_predsender_results(p2, p1, comparison, spec["slug"])
            stub = f"{spec['slug']}_{comparison['slug']}"

            make_p1_figure(p1_rows, FIG_DIR / f"{stub}_p1.png")
            make_p2_figure(p2_rows, FIG_DIR / f"{stub}_p2.png")
            write_fit_table(
                p1_rows[["label", "actual", "predicted", "gap"]],
                f"Predicted vs Actual Effects: Player 1, {comparison['title']}, {spec['title']}",
                f"tab:{stub}_fullpooled_p1_fit",
                TEX_DIR / f"{stub}_p1_fit.tex",
            )
            write_fit_table(
                p2_rows[["label", "actual", "predicted", "gap"]],
                f"Predicted vs Actual Effects: Player 2, {comparison['title']}, {spec['title']}",
                f"tab:{stub}_fullpooled_p2_fit",
                TEX_DIR / f"{stub}_p2_fit.tex",
            )
            write_note(TEX_DIR / f"{stub}_note.tex", spec)
            write_p1_reg_table(p1_models, comparison, spec, TEX_DIR / f"{stub}_p1_regs.tex")
            write_p2_reg_table(
                p2_models,
                comparison,
                spec,
                TEX_DIR / f"{stub}_p2_regs.tex",
                augmented_models=p2_pred_models,
            )

            p1_action_instability = p1_rows[p1_rows["panel"] == "action"][["label", "actual", "predicted", "game_code"]].copy()
            p1_action_instability["player"] = "P1"
            p1_action_instability["comparison_slug"] = comparison["slug"]
            p1_action_instability["comparison_title"] = comparison["title"]
            p1_action_instability["point_label"] = p1_action_instability["label"].replace(
                {
                    "KW share sent": "KW",
                    "UG share sent": "UG",
                    "TG share sent": "TG",
                }
            )
            instability_points[spec["slug"]]["p1_action"].append(p1_action_instability)

            p1_belief_instability = p1_rows[p1_rows["panel"] == "belief"][["label", "actual", "predicted", "game_code"]].copy()
            p1_belief_instability["player"] = "P1"
            p1_belief_instability["comparison_slug"] = comparison["slug"]
            p1_belief_instability["comparison_title"] = comparison["title"]
            p1_belief_instability["point_label"] = p1_belief_instability["label"].replace(
                {
                    "UG beliefs": "UG b",
                    "TG beliefs": "TG b",
                }
            )
            instability_points[spec["slug"]]["p1_belief"].append(p1_belief_instability)

            p2_actual_instability = p2_pred_rows[["label", "actual", "predicted", "game_code"]].copy()
            p2_actual_instability["player"] = "P2"
            p2_actual_instability["comparison_slug"] = comparison["slug"]
            p2_actual_instability["comparison_title"] = comparison["title"]
            p2_actual_instability["point_label"] = p2_actual_instability["label"].replace(
                {
                    "UG acceptance": "UG",
                    "TG share sent": "TG",
                }
            )
            instability_points[spec["slug"]]["p2_actual"].append(p2_actual_instability)

            p2_hyp_instability = p2_hyp_rows[["label", "actual", "predicted", "game_code"]].copy()
            p2_hyp_instability["player"] = "P2"
            p2_hyp_instability["comparison_slug"] = comparison["slug"]
            p2_hyp_instability["comparison_title"] = comparison["title"]
            p2_hyp_instability["point_label"] = p2_hyp_instability["label"].replace(
                {
                    "UG hypothetical action": "UG h",
                    "TG hypothetical action": "TG h",
                }
            )
            instability_points[spec["slug"]]["p2_hyp"].append(p2_hyp_instability)

            pred_stub = f"{stub}_predsender"
            make_p2_figure(p2_pred_rows, FIG_DIR / f"{pred_stub}_p2.png")
            write_fit_table(
                p2_pred_rows[["label", "actual", "predicted", "gap"]],
                f"Predicted vs Actual Effects: Player 2 with Predicted Player 1 Allocation, {comparison['title']}, {spec['title']}",
                f"tab:{pred_stub}_fullpooled_p2_fit",
                TEX_DIR / f"{pred_stub}_p2_fit.tex",
            )
            write_p2_predsender_reg_table(
                p2_pred_models,
                comparison,
                spec,
                TEX_DIR / f"{pred_stub}_p2_regs.tex",
            )

    for spec in SPECIFICATIONS:
        p1_action_points = pd.concat(instability_points[spec["slug"]]["p1_action"], ignore_index=True)
        p1_belief_points = pd.concat(instability_points[spec["slug"]]["p1_belief"], ignore_index=True)
        p2_actual_points = pd.concat(instability_points[spec["slug"]]["p2_actual"], ignore_index=True)
        p2_hyp_points = pd.concat(instability_points[spec["slug"]]["p2_hyp"], ignore_index=True)
        # Emit the figure's full cell data (all four panels) so every predicted/actual
        # pair behind fig:instability_all is log-backed — in particular the four P2
        # hypothetical predicted values, which appear in no fit table (2026-07-20).
        instability_cells = pd.concat(
            [
                points.assign(panel=panel)
                for panel, points in [
                    ("p1_action", p1_action_points),
                    ("p1_belief", p1_belief_points),
                    ("p2_actual", p2_actual_points),
                    ("p2_hyp", p2_hyp_points),
                ]
            ],
            ignore_index=True,
        )
        instability_cells[
            ["panel", "comparison_title", "label", "point_label", "actual", "predicted"]
        ].to_csv(TEX_DIR / f"{spec['slug']}_instability_cells.csv", index=False)
        # 2026-07-19 (SN): the four single-panel exhibits are superseded in the paper
        # by the combined 2x2 panel below; calls kept, commented, for re-enabling.
        # make_instability_figure(
        #     p1_action_points,
        #     "Player 1",
        #     spec,
        #     FIG_DIR / f"{spec['slug']}_instability_p1.png",
        #     title_override=f"Player 1 action, {spec['title']}",
        # )
        # make_instability_figure(
        #     p1_belief_points,
        #     "Player 1",
        #     spec,
        #     FIG_DIR / f"{spec['slug']}_instability_p1_beliefs.png",
        #     title_override=f"Player 1 beliefs, {spec['title']}",
        # )
        # make_instability_figure(
        #     p2_actual_points,
        #     "Player 2",
        #     spec,
        #     FIG_DIR / f"{spec['slug']}_instability_p2.png",
        #     title_override=f"Player 2, {spec['title']} + Fitted Share Sent P1",
        # )
        # make_instability_figure(
        #     p2_hyp_points,
        #     "Player 2",
        #     spec,
        #     FIG_DIR / f"{spec['slug']}_instability_p2_hyp.png",
        #     title_override=f"Player 2 hypothetical action, {spec['title']} + Fitted Share Sent P1",
        # )
        make_instability_panel_figure(
            [
                (p1_action_points, "Player 1 actions"),
                (p1_belief_points, "Player 1 chosen-action beliefs"),
                (p2_actual_points, "Player 2 outcomes"),
                (p2_hyp_points, "Player 2 hypothetical actions"),
            ],
            FIG_DIR / f"{spec['slug']}_instability_2x2.png",
        )
        write_instability_regression_table(
            pd.concat([p1_action_points, p1_belief_points, p2_actual_points, p2_hyp_points], ignore_index=True),
            spec,
            TEX_DIR / f"{spec['slug']}_instability_regs.tex",
        )


if __name__ == "__main__":
    main()
