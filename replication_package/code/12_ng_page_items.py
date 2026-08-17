#!/usr/bin/env python3
"""Page-item computations for the NG revision (2026-07-19).

P8 (p.25-26): companion to Table 2 - mean action by counterpart proximity x category
    (pooled panel + by-game log), output player1_sp_by_category_action.tex.
P7 (p.25): proportion test for the Moral vs Mutual Benefit/Cooperation gap in high SP.
P6 (p.22): representative-quote candidates per game x category and per SP level (control
    sample, editorial selection happens in the paper), output quote_candidates.txt.
    NOTE: the second half of P6 (hypothetical-allocation SP consistency at the person level)
    needs the hp-allocation microdata, which is not in the package (AA has it).

Inputs:  data/player{1,2}_all_categorized.xlsx
Outputs: output/tables/player1_sp_by_category_action.tex, sp_action_stats.txt,
         quote_candidates.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
HIGH_SP = ["Anonymous peer", "Teammate / coworker", "Friend"]  # sp_num >= 2, matches Table 2
GAMES = {"dgkw": "DG-KW", "dglt": "DG-LT", "ug": "UG", "tg": "TG"}
SEED = 42

L = []


def log(*s):
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def main():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p1 = p1[p1.category.isin(CATS)].copy()
    p1["high_sp"] = p1["social_proximity"].isin(HIGH_SP)

    # -- reproduce Table 2's shares as a guard (pooled across games and conditions) --------
    shares = p1.groupby("category")["high_sp"].mean()
    log("High-SP share within category (pooled; must match Table 2: 50.6/8.1/40.6):")
    log((shares * 100).round(1).to_string())
    assert abs(shares["Moral"] * 100 - 50.6) < 0.15
    assert abs(shares["Self-interest"] * 100 - 8.1) < 0.15
    assert abs(shares["Mutual Benefit / Cooperation"] * 100 - 40.6) < 0.15

    # -- P7: Moral vs MBC high-SP proportion test ------------------------------------------
    m = p1[p1.category == "Moral"]["high_sp"]
    c = p1[p1.category == "Mutual Benefit / Cooperation"]["high_sp"]
    z, pv = proportions_ztest([m.sum(), c.sum()], [len(m), len(c)])
    log(f"\nP7 test, high SP Moral ({m.mean():.3f}, N {len(m)}) vs MBC ({c.mean():.3f}, "
        f"N {len(c)}): z = {z:.2f}, p = {pv:.2e}")

    # -- P8: mean action by SP x category --------------------------------------------------
    pooled = (p1.groupby(["high_sp", "category"])["share_sent"].agg(["mean", "size"])
              .unstack("category").reindex(columns=CATS, level=1))
    log("\nMean action (share of budget) by SP x category, pooled:")
    log(pooled.round(3).to_string())
    log("\nBy game (control condition only):")
    ctrl = p1[p1.story == 0]
    for g, glab in GAMES.items():
        d = ctrl[ctrl.game == g]
        if len(d) == 0:
            continue
        tab = d.groupby(["high_sp", "category"])["share_sent"].mean().unstack().reindex(
            columns=CATS)
        log(f"  {glab}:")
        log(tab.round(3).to_string())

    def cell(hi, cat, what):
        return pooled.loc[hi, (what, cat)]

    rows_share, rows_act = [], []
    for hi, lab in [(False, "Low social proximity"), (True, "High social proximity")]:
        srow = [f"{p1[p1.category == c]['high_sp'].eq(hi).mean() * 100:.1f}\\%" for c in CATS]
        # within-category distribution (columns sum to 100), as in Table 2
        srow = [f"{(p1[(p1.category == c)]['high_sp'] == hi).mean() * 100:.1f}\\%" for c in CATS]
        arow = [f"{cell(hi, c, 'mean') * 100:.1f}\\%" for c in CATS]
        rows_share.append(f"{lab} & " + " & ".join(srow) + r" \\")
        rows_act.append(f"{lab} & " + " & ".join(arow) + r" \\")

    tex = r"""\begin{table}[!htbp]
\centering
\caption{\textbf{Player 1: Counterpart Proximity within Stated Reason Category}}
\label{tab:player1_sp_by_category}
\begin{tabular}{lccc}
\toprule
& Moral & Self-interest & Mutual Benefit/Cooperation \\
\midrule
\multicolumn{4}{l}{\emph{Panel A: Distribution of counterpart proximity (column shares)}} \\ % chktex 13
""" + "\n".join(rows_share) + r"""
\midrule
\multicolumn{4}{l}{\emph{Panel B: Mean action, share of the budget}} \\ % chktex 13
""" + "\n".join(rows_act) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Pooled across all games and conditions for Player 1, classified sample. Panel A reports the distribution of counterpart proximity within each stated reason category (columns sum to 100\%). Panel B reports the mean action (share of the budget transferred, offered, or sent) in each proximity-by-category cell. High social proximity: the stated reason refers to the counterpart as at least an anonymous peer (anonymous peer, teammate or coworker, friend); low: no mention of the counterpart or an abstract stranger.
\end{flushleft}
\end{table}
"""
    (TABLES / "player1_sp_by_category_action.tex").write_text(tex)

    # -- P6: quote candidates --------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    out = ["Representative-quote CANDIDATES (control sample; editorial selection by SN).",
           "Rule: within cell, up to 4 random draws among responses of 60-220 characters.", ""]
    p1q = p1[(p1.story == 0)].dropna(subset=["reasons"])
    for g in ["dgkw", "ug", "tg"]:
        for cat in CATS:
            d = p1q[(p1q.game == g) & (p1q.category == cat)]
            d = d[d.reasons.str.len().between(60, 220)]
            take = d.sample(min(4, len(d)), random_state=SEED) if len(d) else d
            out.append(f"--- P1 {GAMES[g]} | {cat} (N cell {len(d)}) ---")
            out += [f"  * {t.strip()}" for t in take.reasons]
            out.append("")
    for hi, lab in [(True, "High SP"), (False, "Low SP")]:
        d = p1q[p1q.high_sp == hi]
        d = d[d.reasons.str.len().between(60, 220)]
        take = d.sample(min(6, len(d)), random_state=SEED + 1)
        out.append(f"--- P1 {lab} (any game) ---")
        out += [f"  * [{GAMES[r.game]}, {r.category}] {r.reasons.strip()}"
                for r in take.itertuples()]
        out.append("")
    p2 = pd.read_excel(DATA / "player2_all_categorized.xlsx")
    p2["story"] = pd.to_numeric(p2["story"], errors="coerce")
    p2q = p2[(p2.story == 0)].dropna(subset=["reasons"])
    for cat in ["Moral good", "Moral bad", "Self-interest", "Mutual Benefit / Cooperation"]:
        d = p2q[p2q.category == cat]
        d = d[d.reasons.str.len().between(60, 220)]
        take = d.sample(min(3, len(d)), random_state=SEED + 2) if len(d) else d
        out.append(f"--- P2 {cat} (N cell {len(d)}) ---")
        out += [f"  * [{r.game.upper()}] {r.reasons.strip()}" for r in take.itertuples()]
        out.append("")
    (TABLES / "quote_candidates.txt").write_text("\n".join(out))

    (TABLES / "sp_action_stats.txt").write_text("\n".join(L))
    print(f"\nwrote player1_sp_by_category_action.tex, sp_action_stats.txt, quote_candidates.txt")


if __name__ == "__main__":
    main()
