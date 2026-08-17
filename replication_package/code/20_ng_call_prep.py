#!/usr/bin/env python3
"""Pre-call robustness checks for the 2026-07-23 methods call with NG.

Three exercises, all diagnostics on existing exhibits (no new paper tables):

A. Between-cell share of action variance with cells = category x tercile of the
   hypothetical belief (UG, TG) -- the belief-inclusive counterpart of the
   between-category share of eq. (7) / tab:heterogeneity_decomposition. Terciles
   computed within game on the pooled four-condition classified sample (common
   breakpoints across conditions). Reported next to the categories-only share on
   the same belief-nonmissing sample, so the comparison isolates the cell
   definition rather than the sample.

B. Prereg-literal variants of the symmetrized Oaxaca of tab:oaxaca_catbelief
   (same decompose() as 11_oaxaca.py):
   B0  validation -- category-only DG-KW must reproduce the published rows;
   B1  categories-only cells in the UG and TG (AsPredicted #286187 wording;
       full classified sample, no belief-nonmissing restriction);
   B2  DG-KW with cells = social-proximity levels (AsPredicted #261370 wording).

C. Per-cell sample sizes for the exact tab:oaxaca_catbelief configuration
   (category x tercile, terciles on the two compared conditions), flagging thin
   (N<10) and one-sided cells.

Inputs:  data/player1_all_categorized.xlsx
Outputs: output/tables/ng_call_prep_stats.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
STORY = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}
COMPARISONS = [("Market vs Control", 1, 0), ("Aid vs Bonus", 4, 2)]

L: list[str] = []


def log(*s: object) -> None:
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def between_share(d: pd.DataFrame, cellcol: str) -> tuple[float, int, int]:
    """Between-cell share of var(share_sent): R^2 on cell dummies, as in 05/E3."""
    y = d.share_sent
    tot = y.var(ddof=1)
    grand = y.mean()
    nb = sum(len(gr) * (gr.mean() - grand) ** 2 for _, gr in d.groupby(cellcol, observed=True).share_sent)
    k = d[cellcol].nunique()
    return nb / (len(y) - 1) / tot, k, len(y)


def decompose(d: pd.DataFrame, treated: int, baseline: int, ref_means: dict) -> tuple[float, float]:
    """Verbatim from 11_oaxaca.py (empty cells -> observed mean to representation)."""
    qT = d[d.story == treated]["cell"].value_counts(normalize=True)
    qB = d[d.story == baseline]["cell"].value_counts(normalize=True)
    yT = d[d.story == treated].groupby("cell")["share_sent"].mean()
    yB = d[d.story == baseline].groupby("cell")["share_sent"].mean()
    rep = beh = 0.0
    for c in sorted(set(qT.index) | set(qB.index)):
        qt, qb = qT.get(c, 0.0), qB.get(c, 0.0)
        yt, yb = yT.get(c, np.nan), yB.get(c, np.nan)
        ybar_ref = ref_means.get(c, np.nanmean([yt, yb]))
        y_avg = np.nanmean([yt, yb])
        rep += (qt - qb) * (ybar_ref if np.isfinite(ybar_ref) else y_avg)
        if np.isfinite(yt) and np.isfinite(yb):
            beh += ref_means["__q__"].get(c, (qt + qb) / 2) * (yt - yb)
    return rep, beh


def symmetrized(d: pd.DataFrame, treated: int, baseline: int) -> tuple[float, float, float]:
    """Symmetrized (Shapley) split; returns (dmean, rep, beh)."""
    dmean = (d[d.story == treated]["share_sent"].mean()
             - d[d.story == baseline]["share_sent"].mean())
    qT = d[d.story == treated]["cell"].value_counts(normalize=True)
    qB = d[d.story == baseline]["cell"].value_counts(normalize=True)
    ref = {"__q__": ((qT + qB) / 2).fillna(qT).fillna(qB).to_dict()}
    rep, beh = decompose(d, treated, baseline, ref)
    return dmean, rep, beh


def main() -> None:
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p1 = p1[p1.story.isin(STORY) & p1.share_sent.notna()].copy()
    cl = p1[p1.category.isin(CATS)].copy()

    # ------------------------------------------------------------------
    log("=== A. Between-cell variance share: categories vs category x belief tercile ===")
    log("Sample: classified, non-missing hypothetical belief; terciles within game on")
    log("the pooled four-condition sample (common breakpoints across conditions).")
    for game in ["ug", "tg"]:
        d = cl[(cl.game == game) & cl.beliefs_hp.notna()].copy()
        d["ter"] = pd.qcut(d.beliefs_hp, 3, labels=["low", "mid", "high"], duplicates="drop")
        d["cell"] = d.category.astype(str) + " x " + d.ter.astype(str)
        for s in [0, 1, 2, 4]:
            ds = d[d.story == s]
            bc, kc, n = between_share(ds, "category")
            bb, kb, _ = between_share(ds, "cell")
            minn = ds.cell.value_counts().min()
            log(f"  {game.upper():4s} {STORY[s]:8s} N={n:4d}  categories only: {bc:.3f} (k={kc})"
                f"  category x tercile: {bb:.3f} (k={kb}, min cell N={minn})")

    # ------------------------------------------------------------------
    log("\n=== A2. Sensitivity: tercile breakpoints for the belief-BCS (three schemes) ===")
    log("pooled-4 = one qcut per game on all four conditions (section A / the footnote);")
    log("two-cond = qcut on the compared pair, as in 11_oaxaca.py (Control/Market from the")
    log("market pair, Bonus/Aid from the story pair); per-cond = qcut within each condition.")
    for game in ["ug", "tg"]:
        d0 = cl[(cl.game == game) & cl.beliefs_hp.notna()].copy()
        schemes: dict[str, dict[int, float]] = {"pooled-4": {}, "two-cond": {}, "per-cond": {}}
        d = d0.copy()
        d["ter"] = pd.qcut(d.beliefs_hp, 3, labels=False, duplicates="drop")
        d["cell"] = d.category.astype(str) + "x" + d.ter.astype(str)
        for s in [0, 1, 2, 4]:
            schemes["pooled-4"][s] = between_share(d[d.story == s], "cell")[0]
        for pair in [(0, 1), (2, 4)]:
            dp = d0[d0.story.isin(pair)].copy()
            dp["ter"] = pd.qcut(dp.beliefs_hp, 3, labels=False, duplicates="drop")
            dp["cell"] = dp.category.astype(str) + "x" + dp.ter.astype(str)
            for s in pair:
                schemes["two-cond"][s] = between_share(dp[dp.story == s], "cell")[0]
        for s in [0, 1, 2, 4]:
            dc = d0[d0.story == s].copy()
            dc["ter"] = pd.qcut(dc.beliefs_hp, 3, labels=False, duplicates="drop")
            dc["cell"] = dc.category.astype(str) + "x" + dc.ter.astype(str)
            schemes["per-cond"][s] = between_share(dc, "cell")[0]
        for s in [0, 1, 2, 4]:
            base = between_share(d0[d0.story == s].assign(cell=lambda t: t.category), "cell")[0]
            vals = "  ".join(f"{k}: {v[s]:.3f} (+{v[s] - base:.3f})" for k, v in schemes.items())
            log(f"  {game.upper():4s} {STORY[s]:8s} categories-only {base:.3f} | {vals}")
    log("  (footnote claim to check: increment over categories-only <= 0.05 under every scheme)")

    # ------------------------------------------------------------------
    log("\n=== B0. Validation: category-only DG-KW reproduces tab:oaxaca_catbelief ===")
    published = {("Market vs Control"): (-0.101, -0.039, 39), ("Aid vs Bonus"): (-0.059, -0.039, 65)}
    for comp, treated, baseline in COMPARISONS:
        d = cl[(cl.game == "dgkw") & cl.story.isin([treated, baseline])].copy()
        d["cell"] = d.category
        dmean, rep, _ = symmetrized(d, treated, baseline)
        want = published[comp]
        ok = abs(dmean - want[0]) < 0.0015 and abs(rep - want[1]) < 0.0015
        log(f"  {comp:>18} DG-KW: dmean {dmean:+.3f} rep {rep:+.3f} ({rep / dmean * 100:.0f}%)"
            f"  [published {want[0]:+.3f} / {want[1]:+.3f} / {want[2]}%]  {'OK' if ok else '***MISMATCH***'}")

    log("\n=== B1. Prereg-literal cells: categories only, UG and TG (#286187 wording) ===")
    log("Full classified sample (no belief-nonmissing restriction), so dmean can differ")
    log("slightly from the published rows.")
    for comp, treated, baseline in COMPARISONS:
        for game, glabel in [("ug", "UG"), ("tg", "TG")]:
            d = cl[(cl.game == game) & cl.story.isin([treated, baseline])].copy()
            d["cell"] = d.category
            dmean, rep, beh = symmetrized(d, treated, baseline)
            pct = rep / dmean * 100 if dmean != 0 else np.nan
            log(f"  {comp:>18} {glabel}: dmean {dmean:+.3f}, repr {rep:+.3f} ({pct:.0f}%), "
                f"behav {beh:+.3f}, N {len(d)}")

    log("\n=== B1b. Categories only on the tab:oaxaca_catbelief sample (belief-nonmissing) ===")
    log("Same sample as the published UG/TG rows, so the gap to Table 14's Repr. % isolates")
    log("the cell definition (dropping the belief terciles), not the sample.")
    for comp, treated, baseline in COMPARISONS:
        for game, glabel in [("ug", "UG"), ("tg", "TG")]:
            d = cl[(cl.game == game) & cl.story.isin([treated, baseline])].dropna(subset=["beliefs_hp"]).copy()
            d["cell"] = d.category
            dmean, rep, beh = symmetrized(d, treated, baseline)
            pct = rep / dmean * 100 if dmean != 0 else np.nan
            log(f"  {comp:>18} {glabel}: dmean {dmean:+.3f}, repr {rep:+.3f} ({pct:.0f}%), "
                f"behav {beh:+.3f}, N {len(d)}")

    log("\n=== B2. Prereg-literal DG-KW: cells = social-proximity levels (#261370 wording) ===")
    for comp, treated, baseline in COMPARISONS:
        d = cl[(cl.game == "dgkw") & cl.story.isin([treated, baseline])
               & cl.social_proximity.notna()].copy()
        d["cell"] = d.social_proximity
        dmean, rep, beh = symmetrized(d, treated, baseline)
        pct = rep / dmean * 100 if dmean != 0 else np.nan
        log(f"  {comp:>18} DG-KW: dmean {dmean:+.3f}, repr {rep:+.3f} ({pct:.0f}%), "
            f"behav {beh:+.3f}, N {len(d)}, levels: {sorted(d.cell.unique())}")

    # ------------------------------------------------------------------
    log("\n=== C. Per-cell N for the exact tab:oaxaca_catbelief cells (11_oaxaca.py config) ===")
    for comp, treated, baseline in COMPARISONS:
        for game, glabel in [("ug", "UG"), ("tg", "TG")]:
            d = cl[(cl.game == game) & cl.story.isin([treated, baseline])].dropna(subset=["beliefs_hp"]).copy()
            d["ter"] = pd.qcut(d.beliefs_hp, 3, labels=["low", "mid", "high"], duplicates="drop")
            d["cell"] = d.category.astype(str) + " x " + d.ter.astype(str)
            tab = d.groupby(["cell", "story"], observed=True).size().unstack(fill_value=0)
            tab = tab.rename(columns=STORY)
            log(f"  {comp} — {glabel}:")
            for cname, r in tab.iterrows():
                flags = []
                if (r < 10).any():
                    flags.append("THIN<10")
                if (r == 0).any():
                    flags.append("ONE-SIDED")
                flag = ("  <-- " + ", ".join(flags)) if flags else ""
                log(f"      {cname:45s} " + "  ".join(f"{c}={int(v):3d}" for c, v in r.items()) + flag)

    # ------------------------------------------------------------------
    log("\n=== D. Table 13 robustness: unclassified answers as their own (fourth) category ===")
    log("Uses only observed labels (no forced classification): every answer with a")
    log("non-substantive or missing category joins one residual group. Reported next to")
    log("the published classified-only between-category share.")
    for game, glabel in [("dgkw", "DG-KW"), ("dglt", "DG-LT"), ("ug", "UG"), ("tg", "TG")]:
        for s in [0, 1, 2, 4]:
            d = p1[(p1.game == game) & (p1.story == s)].copy()
            d["cell4"] = d.category.where(d.category.isin(CATS), "Unclassified")
            b4, k4, n4 = between_share(d, "cell4")
            dc = d[d.category.isin(CATS)].copy()
            dc["cell3"] = dc.category
            b3, _, n3 = between_share(dc, "cell3")
            n_res = n4 - n3
            log(f"  {glabel:5s} {STORY[s]:8s} classified-only: {b3:.3f} (N={n3:4d})  "
                f"with residual group: {b4:.3f} (N={n4:4d}, unclassified={n_res:3d}, "
                f"SD={d.share_sent.std(ddof=1):.3f})")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "ng_call_prep_stats.txt").write_text("\n".join(L) + "\n")
    print(f"\nwrote {TABLES / 'ng_call_prep_stats.txt'}")


if __name__ == "__main__":
    main()
