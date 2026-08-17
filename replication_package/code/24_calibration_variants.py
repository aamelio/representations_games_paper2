#!/usr/bin/env python3
"""Calibration variants matrix (NG meeting 2026-07-23, items 1-2).

Crosses four specification choices on top of 08_calibration.py (imported verbatim):
  - aggregation of the category moments: mean (baseline) / median choices, mean beliefs /
    median everything;
  - TG norm anchor: equal-payoff t*(s) (baseline, amended 2026-07-19) / equal-split 1/2
    (the earlier model's anchor; its predicted TG belief sensitivity has no dt*/ds term);
  - UG belief point feeding the schedule inversion: reference-action belief p_hp at 1/3
    (baseline; chosen-offer belief is the held-out overid moment) / chosen-offer belief
    p_ch at t_ch (then p_hp is held out);
  - TG believed share: hypothetical at the one-third reference (baseline) / chosen-action.

For every variant: (sigma/mu, rho/mu) per category, UG schedule (a,b) + validity, the
MBC joint solve (interior root vs boundary, with the x=0 diagnostic), the held-out
overidentification gap, and predicted belief sensitivities against the control OLS slopes
(estimated slopes are variant-invariant). Two-point diagnostic per aggregation. Bootstrap
SEs (participants resampled within game, B=1000, seed 42) for the two decision-relevant
alternatives: median-choices baseline-anchor and mean equal-split-anchor.

Inputs:  data/player1_all_categorized.xlsx (via 08_calibration.load_control)
Output:  output/tables/calibration_variants_stats.txt
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output" / "tables"

spec = importlib.util.spec_from_file_location("calib08", HERE / "08_calibration.py")
c8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c8)

R, CATS, REF = c8.R, c8.CATS, c8.REF
B_BOOT, SEED = 1000, 42

AGGS = ["mean", "median_choices", "median_all"]
ANCHORS = ["equal_payoff", "equal_split"]
UG_BELIEFS = ["reference", "chosen"]
TG_BELIEFS = ["hp", "chosen"]

CHOICE_KEYS = {"t_dgkw", "t_ug", "t_tg", "t_chosen"}
BELIEF_KEYS = {"p_hp", "p_chosen", "s_hp", "s_chosen"}

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def moments_agg(df: pd.DataFrame, agg: str) -> dict:
    """Category moments under the chosen aggregation (08's layout, mean or median)."""
    def stat(series: pd.Series, key: str) -> float:
        if agg == "mean" or (agg == "median_choices" and key in BELIEF_KEYS):
            return series.mean()
        return series.median()

    out = {}
    for cat in CATS:
        m = {}
        for g in ["dgkw", "ug", "tg"]:
            d = df[(df.game == g) & (df.category == cat)]
            m[f"t_{g}"], m[f"n_{g}"] = stat(d["share_sent"], f"t_{g}"), len(d)
        du = df[(df.game == "ug") & (df.category == cat)]
        dt = df[(df.game == "tg") & (df.category == cat)]
        m["p_hp"] = stat(du["beliefs_hp"].dropna(), "p_hp")
        m["p_chosen"] = stat(du["beliefs"].dropna(), "p_chosen")
        m["t_chosen"] = stat(du.loc[du["beliefs"].notna(), "share_sent"], "t_chosen")
        m["s_hp"] = stat(dt["beliefs_hp"].dropna(), "s_hp")
        m["s_chosen"] = stat(dt["beliefs"].dropna(), "s_chosen")
        out[cat] = m
    return out


def anchor_fns(anchor: str):
    if anchor == "equal_payoff":
        return c8.te, c8.dte
    return (lambda s: 0.5), (lambda s: 0.0)


def foc_ab_general(m: dict, x: float, y: float, ug_belief: str):
    """UG schedule from the FOC plus one belief level point; NaN when the solve degenerates.

    reference: level point p_hp at 1/3, FOC evaluated at t_ug (08's convention);
    chosen:    level point p_ch at t_ch, FOC evaluated at t_ch (subsample-consistent)."""
    if ug_belief == "reference":
        t_eval, t_level, p_level = m["t_ug"], REF, m["p_hp"]
    else:
        t_eval, t_level, p_level = m["t_chosen"], m["t_chosen"], m["p_chosen"]
    tp = t_eval - 0.5
    denom = 2 * tp * x - y + x - t_level * x
    if abs(denom) < 1e-10:
        return np.nan, np.nan
    b = (-tp - p_level * x) / denom
    return p_level - b * t_level, b


def calibrate_variant(m_all: dict, anchor: str, ug_belief: str, tg_belief: str) -> dict:
    te_f, dte_f = anchor_fns(anchor)
    s_key = "s_hp" if tg_belief == "hp" else "s_chosen"

    def locus(m: dict, x: float) -> float:
        return (m["t_tg"] - te_f(m[s_key]) + x * (1 + R) * (1 - m[s_key])) / R

    res = {}
    for cat in ["Moral", "Self-interest"]:
        m = m_all[cat]
        x = max(0.0, 0.5 - m["t_dgkw"])
        y = locus(m, x)
        a, b = foc_ab_general(m, x, y, ug_belief)
        valid = np.isfinite(b) and b > 0 and 0 <= a and a + b <= 1
        res[cat] = dict(x=x, y=y, a=a, b=b, ab_valid=valid)

    si = res["Self-interest"]
    a_c, b_c = si["a"], si["b"]
    m = m_all["Mutual Benefit / Cooperation"]
    mbc = dict(a=a_c, b=b_c, ab_valid=si["ab_valid"], ab_imposed=True)
    if np.isfinite(b_c):
        def gap(x: float) -> float:
            return c8.ug_foc_offer(x, locus(m, x), a_c, b_c) - m["t_ug"]
        from scipy.optimize import brentq
        try:
            x = brentq(gap, 0.0, 0.499)
            mbc["boundary"] = False
        except ValueError:
            grid = np.linspace(0.0, 0.499, 500)
            x = float(grid[np.argmin(np.abs([gap(g) for g in grid]))])
            mbc["boundary"] = True
        mbc.update(x=x, y=locus(m, x), gap_at_0=gap(0.0))
    else:
        mbc.update(x=np.nan, y=np.nan, boundary=True, gap_at_0=np.nan)
    res["Mutual Benefit / Cooperation"] = mbc

    for cat in CATS:
        m, p = m_all[cat], res[cat]
        a_s, b_s = (p["a"], p["b"]) if p["ab_valid"] else (a_c, b_c)
        p["sens_ug_pred"] = (p["x"] / (2 * b_s * p["x"] + 1.0)
                             if np.isfinite(b_s) else np.nan)
        p["sens_tg_pred"] = dte_f(m[s_key]) + (1 + R) * p["x"]
        # held-out overidentification: whichever belief point the inversion did not use
        if np.isfinite(b_s):
            if ug_belief == "reference":
                p["overid_pred"], p["overid_meas"] = a_s + b_s * m["t_chosen"], m["p_chosen"]
                p["overid_moment"] = "p_chosen"
            else:
                p["overid_pred"], p["overid_meas"] = a_s + b_s * REF, m["p_hp"]
                p["overid_moment"] = "p_hp"
        else:
            p["overid_pred"] = p["overid_meas"] = np.nan
            p["overid_moment"] = "--"
        p["t_ug_pred"] = (c8.ug_foc_offer(p["x"], p["y"], a_s, b_s)
                          if np.isfinite(b_s) else np.nan)
        p["t_ug_actual"] = m["t_ug"]
    return res


def fmt(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}" if np.isfinite(v) else "--"


def report_variant(tag: str, res: dict, slopes: dict) -> None:
    log(f"--- {tag} ---")
    for cat in CATS:
        p = res[cat]
        flags = []
        if p.get("ab_imposed"):
            flags.append("SI schedule imposed" + ("" if p["ab_valid"] else " (INVALID here)"))
            if p.get("boundary"):
                flags.append(f"BOUNDARY x=0 (implied-actual UG offer at x=0: "
                             f"{fmt(p.get('gap_at_0', np.nan))})")
            else:
                flags.append("interior root")
        elif not p["ab_valid"]:
            flags.append("schedule unidentified (b<=0 or bounds)")
        su, st = slopes[("ug", cat)][0], slopes[("tg", cat)][0]
        ru = abs(su) / p["sens_ug_pred"] if p["sens_ug_pred"] else np.inf
        rt = st / p["sens_tg_pred"] if p["sens_tg_pred"] else np.inf
        log(f"  {cat:<30} x={fmt(p['x'])}  y={fmt(p['y'])}  "
            f"(a,b)=({fmt(p['a'])},{fmt(p['b'])})  "
            f"overid[{p['overid_moment']}] {fmt(p['overid_pred'])} vs {fmt(p['overid_meas'])} "
            f"(gap {fmt(p['overid_meas'] - p['overid_pred'])})")
        log(f"  {'':<30} sens UG pred {fmt(p['sens_ug_pred'])} (obs {su:+.3f}, ratio "
            f"{fmt(ru, 2)})  TG pred {fmt(p['sens_tg_pred'])} (obs {st:+.3f}, ratio "
            f"{fmt(rt, 2)})" + ("   [" + "; ".join(flags) + "]" if flags else ""))
    log("")


def bootstrap_variant(df: pd.DataFrame, agg: str, anchor: str,
                      ug_belief: str, tg_belief: str):
    rng = np.random.default_rng(SEED)
    keys = ["x", "y", "a", "b"]
    idx_by_game = {g: df.index[df.game == g].to_numpy() for g in ["dgkw", "ug", "tg"]}
    draws = []
    for _ in range(B_BOOT):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_game.values()])
        try:
            r = calibrate_variant(moments_agg(df.loc[take], agg), anchor,
                                  ug_belief, tg_belief)
            draws.append({(c, k): r[c].get(k, np.nan) for c in CATS for k in keys})
        except Exception:
            continue
    bd = pd.DataFrame(draws)
    return bd.std(), len(bd)


def main() -> None:
    df = c8.load_control()
    slopes = c8.empirical_slopes(df)

    log("Calibration variants matrix (NG meeting 2026-07-23, items 1-2)")
    log(f"Baseline = mean / equal_payoff / reference / hp; reproduced first.\n")

    for agg in AGGS:
        m_all = moments_agg(df, agg)
        log(f"=== moments, aggregation: {agg} ===")
        log(pd.DataFrame(m_all).T.round(4).to_string(), "")
        a2 = {c: c8.two_point_ab(m_all[c]) for c in CATS}
        log("two-point diagnostic (a,b): "
            + "; ".join(f"{c.split()[0]} ({fmt(a2[c][0])},{fmt(a2[c][1])})" for c in CATS),
            "")

    base = calibrate_variant(moments_agg(df, "mean"), "equal_payoff", "reference", "hp")
    b08 = c8.calibrate(c8.moments(df))
    same = all(abs(base[c]["x"] - b08[c]["x"]) < 1e-12
               and abs(base[c]["y"] - b08[c]["y"]) < 1e-12 for c in CATS)
    log(f"baseline reproduction against 08_calibration.calibrate: "
        f"{'EXACT' if same else 'MISMATCH -- do not trust the grid'}\n")

    log("=== full grid ===\n")
    for agg in AGGS:
        m_all = moments_agg(df, agg)
        for anchor in ANCHORS:
            for ug_b in UG_BELIEFS:
                for tg_b in TG_BELIEFS:
                    res = calibrate_variant(m_all, anchor, ug_b, tg_b)
                    report_variant(f"agg={agg} anchor={anchor} ugB={ug_b} tgB={tg_b}",
                                   res, slopes)

    log("=== bootstrap SEs, decision-relevant alternatives (B=1000, seed 42) ===")
    for tag, agg, anchor in [("median_choices / equal_payoff / reference / hp",
                              "median_choices", "equal_payoff"),
                             ("mean / equal_split / reference / hp", "mean", "equal_split")]:
        ses, n_ok = bootstrap_variant(df, agg, anchor, "reference", "hp")
        log(f"{tag}  ({n_ok}/{B_BOOT} successful draws)")
        for cat in CATS:
            log(f"  {cat:<30} se(x)={fmt(ses.get((cat, 'x'), np.nan))} "
                f"se(y)={fmt(ses.get((cat, 'y'), np.nan))} "
                f"se(a)={fmt(ses.get((cat, 'a'), np.nan))} "
                f"se(b)={fmt(ses.get((cat, 'b'), np.nan))}")
    log("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_variants_stats.txt").write_text("\n".join(LOG))
    print(f"\nwrote {OUT / 'calibration_variants_stats.txt'}")


if __name__ == "__main__":
    main()
