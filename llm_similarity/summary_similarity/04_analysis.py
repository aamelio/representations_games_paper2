#!/usr/bin/env python3
"""Task 7 — analyses on the production summary-similarity scores.

Use 1 (validation): Aid vs Bonus mean emphasis scores per aspect, pooled and by game,
with Welch tests; dominant-category shares; the cross-game placebo (summaries precede
the game, so same-story scores should not differ materially across games).

Use 2 (heterogeneity): merge with the classified analysis data via PROLIFIC_PID and ask
NG's question — within a story condition, does a participant's summary emphasis predict
(a) the stated reason category, (b) the action, (c) beliefs, with the TG-vs-DG contrast.

Inputs:  production_scores.csv, ../../replication_package/data/player1_all_categorized.xlsx
Output:  out/summary_similarity_stats.txt (aggregates only; no participant text)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "replication_package" / "data"
OUT = HERE / "out"

ASPECTS = ["F", "O", "J"]
ASPECT_CAT = {"F": "Moral", "O": "Self-interest", "J": "Mutual Benefit / Cooperation"}
GAMES = ["dgkw", "dglt", "ug", "tg"]

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def validation(sc: pd.DataFrame) -> None:
    log("=== USE 1: validation (Aid vs Bonus emphasis in participants' own summaries) ===\n")
    log("pooled means (0-100):")
    log(sc.groupby("story")[ASPECTS].mean().round(1).to_string())
    log("\nWelch tests, Aid - Bonus per aspect (pooled):")
    for a in ASPECTS:
        x = sc.loc[sc.story == "aid", a]
        y = sc.loc[sc.story == "bonus", a]
        t, p = stats.ttest_ind(x, y, equal_var=False)
        log(f"  {a} ({ASPECT_CAT[a]}): diff {x.mean() - y.mean():+.1f}  t={t:.1f}  p={p:.2g}")
    log("\nby game x story:")
    log(sc.groupby(["game", "story"])[ASPECTS].mean().round(1).to_string())
    log("\ndominant shares by story:")
    log(sc.groupby("story")["dominant"].value_counts(normalize=True).round(3).to_string())
    log("\ncross-game placebo (same story, means across games; summaries precede the game):")
    for story in ["aid", "bonus"]:
        d = sc[sc.story == story]
        rng = d.groupby("game")[ASPECTS].mean().agg(lambda c: c.max() - c.min())
        log(f"  {story}: max-min across games  F {rng['F']:.1f}  O {rng['O']:.1f}  J {rng['J']:.1f}")
    log("")


def heterogeneity(sc: pd.DataFrame) -> None:
    log("=== USE 2: heterogeneity (within story condition, pre-game emphasis -> behavior) ===\n")
    cat = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    log("categorized-file story codes present:",
        dict(cat["story"].value_counts().sort_index()))
    m = sc.merge(cat, on="PROLIFIC_PID", suffixes=("", "_cat"))
    m = m[m["game"] == m["game_cat"]] if "game_cat" in m.columns else m
    log(f"merged N = {len(m)} of {len(sc)} scored summaries\n")

    log("--- (a) summary dominant aspect vs stated reason category (classified only) ---")
    sub = m[m["category"].isin(ASPECT_CAT.values())]
    tab = pd.crosstab(sub["dominant"], sub["category"], normalize="index").round(3)
    log(tab.to_string())
    match_map = {"F": "Moral", "O": "Self-interest", "J": "Mutual Benefit / Cooperation"}
    sub2 = sub[sub["dominant"].isin(match_map)]
    rate = (sub2["dominant"].map(match_map) == sub2["category"]).mean()
    n_dom = len(sub2)
    chance = (sub2["dominant"].map(match_map).value_counts(normalize=True)
              * sub["category"].value_counts(normalize=True)).sum()
    log(f"\n  dominant-matches-category rate: {rate:.1%} (N={n_dom}; chance approx {chance:.1%})\n")

    log("--- (b) action on standardized summary scores, within story x game (OLS, HC1) ---")
    for g in GAMES:
        for story in ["aid", "bonus"]:
            d = m[(m.game == g) & (m.story == story)].dropna(subset=["share_sent"])
            if len(d) < 50:
                continue
            X = d[ASPECTS].apply(lambda c: (c - c.mean()) / c.std())
            fit = sm.OLS(d["share_sent"].astype(float), sm.add_constant(X)).fit(cov_type="HC1")
            cells = "  ".join(
                f"{a} {fit.params[a]:+.3f} (p={fit.pvalues[a]:.2g})" for a in ASPECTS)
            log(f"  {g:5} {story:5} N={len(d):3}: {cells}")
    log("\n  (coefficients = change in action share per 1 SD of summary emphasis)")

    log("\n--- (c) beliefs (UG acceptance / TG return at reference) on summary scores ---")
    for g in ["ug", "tg"]:
        d = m[(m.game == g)].dropna(subset=["beliefs_hp"])
        d = d[d["story"].isin(["aid", "bonus"])]
        if len(d) < 50:
            continue
        X = d[ASPECTS].apply(lambda c: (c - c.mean()) / c.std())
        X["aid"] = (d["story"] == "aid").astype(float)
        fit = sm.OLS(d["beliefs_hp"].astype(float), sm.add_constant(X)).fit(cov_type="HC1")
        cells = "  ".join(
            f"{a} {fit.params[a]:+.3f} (p={fit.pvalues[a]:.2g})" for a in ASPECTS)
        log(f"  {g:2} N={len(d):4} (story-pooled, aid dummy): {cells}")

    log("\n--- NG's sharpest cell: J (joint-gains) emphasis and the action, TG vs DG-KW ---")
    for g in ["dgkw", "tg"]:
        d = m[(m.game == g) & m["story"].isin(["aid", "bonus"])].dropna(subset=["share_sent"])
        hi = d[d["J"] >= d["J"].median()]
        lo = d[d["J"] < d["J"].median()]
        t, p = stats.ttest_ind(hi["share_sent"], lo["share_sent"], equal_var=False)
        log(f"  {g:5}: action mean, high-J {hi['share_sent'].mean():.3f} vs "
            f"low-J {lo['share_sent'].mean():.3f}  (diff {hi['share_sent'].mean() - lo['share_sent'].mean():+.3f}, p={p:.2g})")


def main() -> None:
    sc = pd.read_csv(HERE / "production_scores.csv")
    log(f"production scores: N={len(sc)}\n")
    validation(sc)
    heterogeneity(sc)
    OUT.mkdir(exist_ok=True)
    (OUT / "summary_similarity_stats.txt").write_text("\n".join(LOG))
    print(f"\nwrote {OUT / 'summary_similarity_stats.txt'}")


if __name__ == "__main__":
    main()
