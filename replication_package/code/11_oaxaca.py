#!/usr/bin/env python3
"""Preregistered symmetrized Blinder-Oaxaca decomposition with category x belief cells.

The table reports two parallel definitions of the belief component of a representation:
  * stated category x quantile group of the hypothetical belief at the common one-third action;
  * stated category x quantile group of the chosen-action ("actual") belief.

Belief groups are computed within game on the two conditions being compared, with tied
values kept together. DG-KW has no belief elicitation, so both specifications use the same
category-only cells. For each comparison (Market vs Control, Aid vs Bonus) and game, the
mean-action difference decomposes into:

  representation component  =  sum_c dq_c * ybar_c
  behavior component        =  sum_c q_c * dybar_c

The symmetrized/Shapley version averages the two paths and is the preregistered variant.
Cells empty in one condition contribute their observed-side mean to the representation
component and zero to the behavior component.

The hypothetical-belief specification retains the previous Table 14 construction exactly:
strategic-game observations with missing hypothetical beliefs are excluded. The actual-
belief specification is estimated on that same legacy sample so the two decompositions have
the same mean difference and N and differ only in which belief defines the cells.

Participant-level nonparametric bootstrap: B = 1000, seed 20260721, resampling within
game x condition strata; belief groups and both decompositions are recomputed in every draw.
Standard errors are displayed under the mean difference and every level component.
Representation shares stay point estimates; the Aid-vs-Bonus TG shares are suppressed
because the total effect is indistinguishable from zero.

Inputs:  data/player1_all_categorized.xlsx
Outputs: output/tables/oaxaca_catbelief.tex, oaxaca_catbelief_stats.txt
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
COMPARISONS = [("Market vs Control", 1, 0), ("Aid vs Bonus", 4, 2)]
GAMES = [("dgkw", "DG-KW"), ("ug", "UG"), ("tg", "TG")]
SPECS = [
    ("hp", "beliefs_hp", "hypothetical beliefs"),
    ("actual", "beliefs", "actual beliefs"),
]

B = 1000
SEED = 20260721

L = []


def log(*s):
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def two_cond_cells(dg, treated, baseline, game, belief_col):
    """Return the common two-condition sample with the requested representation cells."""
    two = dg[dg.story.isin([treated, baseline])].copy()
    if game == "dgkw":
        two["cell"] = two["category"].astype(str)
        return two

    # Preserve the original Table 14 sample for both specifications.
    two = two.dropna(subset=["beliefs_hp"])
    assert two[belief_col].notna().all(), f"{belief_col} must be complete in {game}"
    try:
        two["belief_group"] = pd.qcut(
            two[belief_col],
            3,
            labels=["low", "mid", "high"],
            duplicates="drop",
        )
    except ValueError:
        two["belief_group"] = "all"
    two["cell"] = (
        two["category"].astype(str) + " x " + two["belief_group"].astype(str)
    )
    return two


def moments(two, treated, baseline):
    """Cell shares and conditional means by condition, plus the mean difference."""
    q_treated = two[two.story == treated]["cell"].value_counts(normalize=True)
    q_baseline = two[two.story == baseline]["cell"].value_counts(normalize=True)
    y_treated = two[two.story == treated].groupby("cell")["share_sent"].mean()
    y_baseline = two[two.story == baseline].groupby("cell")["share_sent"].mean()
    diff = (
        two[two.story == treated]["share_sent"].mean()
        - two[two.story == baseline]["share_sent"].mean()
    )
    return q_treated, q_baseline, y_treated, y_baseline, diff


def symmetrized(dg, treated, baseline, game, belief_col):
    """Return the exact symmetrized decomposition for one cell definition."""
    two = two_cond_cells(dg, treated, baseline, game, belief_col)
    q_treated, q_baseline, y_treated, y_baseline, diff = moments(
        two, treated, baseline
    )
    cells = sorted(set(q_treated.index) | set(q_baseline.index))

    representation = behavior = 0.0
    for cell in cells:
        qt = q_treated.get(cell, 0.0)
        qb = q_baseline.get(cell, 0.0)
        yt = y_treated.get(cell, np.nan)
        yb = y_baseline.get(cell, np.nan)
        average_conditional_mean = np.nanmean([yt, yb])
        representation += (qt - qb) * average_conditional_mean
        if np.isfinite(yt) and np.isfinite(yb):
            behavior += ((qt + qb) / 2) * (yt - yb)

    assert abs(representation + behavior - diff) < 1e-9, (
        "symmetrized decomposition must be exact"
    )
    return {
        "diff": diff,
        "n": len(two),
        "representation": representation,
        "behavior": behavior,
    }


def prepare_data():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    for col in ["share_sent", "beliefs", "beliefs_hp"]:
        p1[col] = pd.to_numeric(p1[col], errors="coerce")
    p1 = p1[p1.category.isin(CATS)].copy()

    strategic = p1.game.isin(["ug", "tg"])
    at_reference = np.isclose(p1["share_sent"], 1 / 3)
    missing_hp = strategic & p1["beliefs_hp"].isna()
    assert (missing_hp == (strategic & at_reference)).all(), (
        "hypothetical beliefs must be missing exactly at the one-third reference action"
    )
    assert p1.loc[strategic, "beliefs"].notna().all()
    log(
        "Legacy hypothetical-belief exclusions at the one-third action:",
        f"UG {int((missing_hp & p1.game.eq('ug')).sum())},",
        f"TG {int((missing_hp & p1.game.eq('tg')).sum())}.",
    )
    return p1


def main():
    p1 = prepare_data()
    order = [
        (comparison, treated, baseline, game, game_label)
        for comparison, treated, baseline in COMPARISONS
        for game, game_label in GAMES
    ]

    # Point estimates
    rows = {}
    for comparison, treated, baseline, game, game_label in order:
        dg = p1[p1.game == game]
        results = {
            slug: symmetrized(dg, treated, baseline, game, belief_col)
            for slug, belief_col, label in SPECS
        }
        assert results["hp"]["n"] == results["actual"]["n"]
        assert abs(results["hp"]["diff"] - results["actual"]["diff"]) < 1e-12
        rows[(comparison, game_label)] = results

        log(
            f"{comparison:>18} {game_label:>6}: "
            f"observed diff {results['hp']['diff']:+.3f}, N {results['hp']['n']}"
        )
        for slug, belief_col, label in SPECS:
            result = results[slug]
            share = (
                result["representation"] / result["diff"] * 100
                if abs(result["diff"]) > 1e-9
                else np.nan
            )
            log(
                f"      {label:>20}: representation "
                f"{result['representation']:+.3f} ({share:.0f}%), "
                f"behavior {result['behavior']:+.3f} ({100 - share:.0f}%), "
                f"residual "
                f"{result['diff'] - result['representation'] - result['behavior']:+.3f}"
            )

    # Participant-level bootstrap
    log("")
    log(
        f"Participant-level bootstrap: B={B}, seed={SEED}, resampling within "
        "game x condition strata; belief groups and both decompositions recomputed per draw."
    )
    rng = np.random.default_rng(SEED)
    game_indices = {}
    for game, game_label in GAMES:
        dg = p1[p1.game == game]
        story_groups = {
            story: dg.index[dg.story == story].to_numpy()
            for story in sorted(dg.story.dropna().unique())
        }
        game_indices[game] = (dg, story_groups)

    metrics = (
        "diff",
        "hp_representation",
        "hp_behavior",
        "hp_share",
        "actual_representation",
        "actual_behavior",
        "actual_share",
    )
    stats = {
        (comparison, game_label): {metric: [] for metric in metrics}
        for comparison, treated, baseline, game, game_label in order
    }

    start = time.perf_counter()
    for _ in range(B):
        for game, game_label in GAMES:
            dg_full, story_groups = game_indices[game]
            picks = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in story_groups.values()
                ]
            )
            bootstrap_game = dg_full.loc[picks]
            for comparison, treated, baseline in COMPARISONS:
                results = {
                    slug: symmetrized(
                        bootstrap_game, treated, baseline, game, belief_col
                    )
                    for slug, belief_col, label in SPECS
                }
                assert abs(results["hp"]["diff"] - results["actual"]["diff"]) < 1e-12
                cell_stats = stats[(comparison, game_label)]
                cell_stats["diff"].append(results["hp"]["diff"])
                for slug, belief_col, label in SPECS:
                    result = results[slug]
                    cell_stats[f"{slug}_representation"].append(
                        result["representation"]
                    )
                    cell_stats[f"{slug}_behavior"].append(result["behavior"])
                    cell_stats[f"{slug}_share"].append(
                        result["representation"] / result["diff"]
                        if abs(result["diff"]) > 1e-9
                        else np.nan
                    )
    runtime = time.perf_counter() - start

    def se_ci(values):
        values = np.asarray(values, dtype=float)
        return (
            np.nanstd(values, ddof=1),
            np.nanpercentile(values, 2.5),
            np.nanpercentile(values, 97.5),
        )

    bootstrap_se = {}
    log("")
    log(
        f"{'Comparison':>18} {'Game':>6} | metric                  "
        f"{'SE':>9} {'CI2.5':>9} {'CI97.5':>9}"
    )
    for comparison, treated, baseline, game, game_label in order:
        cell_stats = stats[(comparison, game_label)]
        bootstrap_se[(comparison, game_label)] = {}
        for metric in (
            "diff",
            "hp_representation",
            "hp_behavior",
            "actual_representation",
            "actual_behavior",
        ):
            se, low, high = se_ci(cell_stats[metric])
            bootstrap_se[(comparison, game_label)][metric] = se
            log(
                f"{comparison:>18} {game_label:>6} | {metric:<23} "
                f"{se:9.4f} {low:+9.4f} {high:+9.4f}"
            )

    log("")
    log("Representation-share diagnostics (representation/difference across draws):")
    log(
        f"{'Comparison':>18} {'Game':>6} {'Belief':>8} | {'median%':>9} "
        f"{'p2.5%':>10} {'p97.5%':>10} {'%degenerate':>12}"
    )
    for comparison, treated, baseline, game, game_label in order:
        for slug, belief_col, label in SPECS:
            shares = np.asarray(
                stats[(comparison, game_label)][f"{slug}_share"], dtype=float
            )
            degenerate = (
                np.isnan(shares) | (shares < 0.0) | (shares > 1.5)
            )
            log(
                f"{comparison:>18} {game_label:>6} {slug:>8} | "
                f"{np.nanmedian(shares) * 100:9.1f} "
                f"{np.nanpercentile(shares, 2.5) * 100:10.1f} "
                f"{np.nanpercentile(shares, 97.5) * 100:10.1f} "
                f"{degenerate.mean() * 100:12.1f}"
            )
    log("")
    log(f"Bootstrap wall-clock: {runtime:.2f} s")

    # LaTeX table
    def f3(value):
        return f"{value:+.3f}"

    def fse(value):
        return f"({value:.3f})"

    dash_key = ("Aid vs Bonus", "TG")
    body = []
    for comparison, treated, baseline, game, game_label in order:
        results = rows[(comparison, game_label)]
        ses = bootstrap_se[(comparison, game_label)]
        hp = results["hp"]
        actual = results["actual"]
        hp_share = (
            "---"
            if (comparison, game_label) == dash_key
            else f"{hp['representation'] / hp['diff'] * 100:.0f}\\%"
        )
        actual_share = (
            "---"
            if (comparison, game_label) == dash_key
            else f"{actual['representation'] / actual['diff'] * 100:.0f}\\%"
        )
        body.append(
            f"{comparison} & {game_label} & {f3(hp['diff'])} & "
            f"{f3(hp['representation'])} & {f3(hp['behavior'])} & {hp_share} & "
            f"{f3(actual['representation'])} & {f3(actual['behavior'])} & "
            f"{actual_share} & {hp['n']} \\\\"
        )
        body.append(
            f" & & {fse(ses['diff'])} & "
            f"{fse(ses['hp_representation'])} & {fse(ses['hp_behavior'])} & & "
            f"{fse(ses['actual_representation'])} & "
            f"{fse(ses['actual_behavior'])} & & \\\\"
        )
        body.append(r"\addlinespace[2pt]")
    body = body[:-1]

    tg_aid = rows[dash_key]["hp"]
    tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Symmetrized Decomposition of Treatment Effects: Hypothetical and Actual Beliefs}}
\label{tab:oaxaca_catbelief}
\begin{tabular}{ll c ccc ccc r}
\toprule
& & & \multicolumn{3}{c}{Hypothetical beliefs} & \multicolumn{3}{c}{Actual beliefs} & \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Comparison & Game & $\Delta$ mean & Repr. & Behav. & Repr. \% &
Repr. & Behav. & Repr. \% & $N$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Symmetrized (Shapley/Oaxaca--Blinder) decomposition of the difference in mean Player 1 actions between conditions into a component due to the distribution of representation cells and a component due to behavior conditional on the cell. In the ultimatum and trust games, cells interact the stated category with a quantile group of the belief; cutoffs are computed within game on the two conditions compared, with tied values kept together. The first specification retains the previous Table 14 construction and uses hypothetical beliefs at the common one-third action. The second uses actual (chosen-action) beliefs; for trust-game senders who chose zero, this belief refers to the hypothetical case of sending one sixth. Both specifications use the same classified sample with non-missing hypothetical beliefs, so the mean difference and $N$ are common across columns; the actual-belief specification is deliberately restricted to this sample. Participants choosing exactly one third are not asked the hypothetical question separately and remain excluded, as in the previous specification. The dictator game has no belief elicitation, so both specifications use categories alone and are identical. Actual beliefs condition on an endogenous chosen action; their decomposition is therefore descriptive and does not isolate an upstream representation channel. Cells empty in one condition contribute their observed mean to the representation component. Bootstrap standard errors in parentheses ($B=1{,}000$ participant-level draws within game $\times$ condition strata; belief groups and both decompositions recomputed in each draw). Representation shares are point estimates; they are not reported for the trust-game story comparison, where the total effect is """ + f3(tg_aid["diff"]) + r""" (bootstrap SE """ + f"{bootstrap_se[dash_key]['diff']:.3f}" + r""").
\end{flushleft}
\end{table}
"""
    (TABLES / "oaxaca_catbelief.tex").write_text(tex)
    (TABLES / "oaxaca_catbelief_stats.txt").write_text("\n".join(L))
    print(f"\nwrote {TABLES / 'oaxaca_catbelief.tex'} and oaxaca_catbelief_stats.txt")


if __name__ == "__main__":
    main()
