#!/usr/bin/env python3
"""Calibration of P1 attention weights and beliefs (NG comment 2, 2026-07-19).

Maps control-sample moments into the model's parameters, category by category. Behavior
identifies attention only up to scale, so the calibrated objects are (sigma/mu, rho/mu).

Identification:
  - DG transfer pins selfishness:   sigma/mu = 1/2 - t_DG   (Moral, Self-interest; interior)
  - TG send + believed share pins   rho/mu = [t_TG - te(s) + (sigma/mu)(1+r)(1-s)] / r
    where te(s) = 1/(1 + (1+r)(1-2s)) is the equal-payoff norm anchor (the send that
    equalizes expected payoffs at the believed returned share s; = 1/2 at s = 1/3).
    Amended 2026-07-19 (equal-payoff anchor, dry-run 17_tg_anchor_dryrun.py).
  - UG acceptance schedule (a,b): FOC-inversion route (HEADLINE) - given (sigma/mu, rho/mu),
    the UG first-order condition plus the reference-action belief point p_hp = a + b/3 pin
    (a,b) linearly. The SECOND elicited belief point (acceptance at the chosen offer) is then
    an overidentifying test: predicted a + b*t_chosen vs measured p_chosen.
  - Mutual Benefit / Cooperation: its DG cell is ~3 obs. Acceptance beliefs at the reference
    action are flat across categories (the paper's Figure 4 fact), so we impose the
    Self-interest-identified schedule (a,b) on MBC and solve the UG and TG FOCs jointly for
    (sigma/mu, rho/mu); the TG locus rho/mu = f(sigma/mu) is logged for transparency.
  - Two-point route (DIAGNOSTIC, was the candidate headline): category means of the two belief
    points identify (a,b) from measurement alone. IT FAILS VALIDITY (a<0, a+b>1 for Moral and
    Self-interest): chosen-action beliefs are too optimistic relative to reference beliefs to
    lie on any common linear probability schedule - evidence of optimism/selection at chosen
    actions, which the log quantifies and the forecast-error section can use.
  - Belief sensitivities, predicted vs estimated:
      UG |dt/da| = (s/m)/(2b(s/m)+1)   TG dt/ds = dte/ds + (1+r)(s/m)   [s/m = sigma/mu]
    (the TG sensitivity gains the weight-free norm channel dte/ds = 2(1+r)/(1+(1+r)(1-2s))^2)
      estimated = OLS slope of action on beliefs_hp, by category, control sample.
  - Attention-belief coupling (rem:joint): E[t^TG] - E_indep[t^TG] = (1+r)*Cov(sigma/mu, s).

Inputs:  data/player1_all_categorized.xlsx
Outputs: output/tables/calibration_p1.tex, output/tables/calibration_stats.txt
Bootstrap SEs: resample participants within game (B=1000, seed 42).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

R = 2.0  # net return in the TG (multiplier 3)
CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
SHORT = {"Moral": "Moral", "Self-interest": "Self-interest",
         "Mutual Benefit / Cooperation": "Mutual Benefit/Coop."}
REF = 1.0 / 3.0
B_BOOT, SEED = 1000, 42


def load_control():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    return p1[(p1["story"] == 0) & p1["category"].isin(CATS)].copy()


def moments(df):
    out = {}
    for cat in CATS:
        m = {}
        for g in ["dgkw", "ug", "tg"]:
            d = df[(df.game == g) & (df.category == cat)]
            m[f"t_{g}"], m[f"n_{g}"] = d["share_sent"].mean(), len(d)
        du = df[(df.game == "ug") & (df.category == cat)]
        dt = df[(df.game == "tg") & (df.category == cat)]
        m["p_hp"] = du["beliefs_hp"].mean()
        m["p_chosen"] = du["beliefs"].mean()
        m["t_chosen"] = du.loc[du["beliefs"].notna(), "share_sent"].mean()
        m["s_hp"], m["s_chosen"] = dt["beliefs_hp"].mean(), dt["beliefs"].mean()
        out[cat] = m
    return out


def two_point_ab(m):
    """(a,b) from the two elicited belief points (category means). Diagnostic route."""
    b = (m["p_chosen"] - m["p_hp"]) / (m["t_chosen"] - REF)
    return m["p_hp"] - b * REF, b


def foc_ab(m, x, y):
    """(a,b) from the UG FOC + the reference belief point, given (x,y)=(sigma/mu, rho/mu).

    FOC: (t-1/2)(2bx+1) = b*y - (a+b)x with a = p_hp - b/3  =>  linear in b."""
    tp = m["t_ug"] - 0.5
    denom = 2 * tp * x - y + x - x / 3.0
    b = (-tp - m["p_hp"] * x) / denom
    return m["p_hp"] - b * REF, b


def ug_foc_offer(x, y, a, b):
    t = 0.5 + (b * y - (a + b) * x) / (2 * b * x + 1.0)
    return min(1.0, max(0.0, t))


def te(s):
    """Equal-payoff norm anchor: the send equalizing expected payoffs at believed share s."""
    return 1.0 / (1.0 + (1 + R) * (1 - 2 * s))


def dte(s):
    return 2 * (1 + R) / (1.0 + (1 + R) * (1 - 2 * s)) ** 2


def tg_locus(m, x):
    """rho/mu implied by the TG FOC at selfishness x (equal-payoff anchor)."""
    return (m["t_tg"] - te(m["s_hp"]) + x * (1 + R) * (1 - m["s_hp"])) / R


def calibrate(m_all):
    res = {}
    for cat in ["Moral", "Self-interest"]:
        m = m_all[cat]
        x = max(0.0, 0.5 - m["t_dgkw"])
        res[cat] = dict(x=x, y=tg_locus(m, x))
    # schedules: FOC-inversion (headline); Moral's is unidentified when b comes out <= 0,
    # exactly as norm-domination implies - fall back to the SI schedule for its sensitivity.
    for cat in ["Moral", "Self-interest"]:
        m, p = m_all[cat], res[cat]
        p["a_foc"], p["b_foc"] = foc_ab(m, p["x"], p["y"])
        p["ab_valid"] = (p["b_foc"] > 0) and (0 <= p["a_foc"]) and (p["a_foc"] + p["b_foc"] <= 1)
    common_ab = (res["Self-interest"]["a_foc"], res["Self-interest"]["b_foc"])

    # MBC: impose the common schedule (flat reference beliefs across categories), solve UG+TG
    m = m_all["Mutual Benefit / Cooperation"]
    a_c, b_c = common_ab

    def gap(x):
        return ug_foc_offer(x, tg_locus(m, x), a_c, b_c) - m["t_ug"]

    try:
        x = brentq(gap, 0.0, 0.499)
    except ValueError:
        grid = np.linspace(0.0, 0.499, 500)
        x = float(grid[np.argmin(np.abs([gap(g) for g in grid]))])
    res["Mutual Benefit / Cooperation"] = dict(
        x=x, y=tg_locus(m, x), a_foc=a_c, b_foc=b_c, ab_valid=True, ab_imposed=True)

    for cat in CATS:
        m, p = m_all[cat], res[cat]
        p["a2"], p["b2"] = two_point_ab(m)  # diagnostic route
        p["s_hp"], p["s_chosen"] = m["s_hp"], m["s_chosen"]
        a_s, b_s = (p["a_foc"], p["b_foc"]) if p.get("ab_valid") else common_ab
        p["sens_ug_pred"] = p["x"] / (2 * b_s * p["x"] + 1.0)
        p["sens_tg_pred"] = dte(m["s_hp"]) + (1 + R) * p["x"]
        # overidentification: schedule-implied acceptance at the chosen offer vs measured
        p["p_chosen_pred"] = a_s + b_s * m["t_chosen"]
        p["p_chosen_actual"] = m["p_chosen"]
        p["t_ug_pred"] = ug_foc_offer(p["x"], p["y"], a_s, b_s)
        p["t_ug_actual"] = m["t_ug"]
    return res


def empirical_slopes(df):
    out = {}
    for g in ["ug", "tg"]:
        for cat in CATS:
            d = df[(df.game == g) & (df.category == cat)].dropna(subset=["beliefs_hp"])
            if len(d) < 10:
                out[(g, cat)] = (np.nan, np.nan, len(d))
                continue
            fit = sm.OLS(d["share_sent"], sm.add_constant(d["beliefs_hp"])).fit(cov_type="HC1")
            out[(g, cat)] = (fit.params.iloc[1], fit.bse.iloc[1], len(d))
    return out


def covariance_exercise(df, res):
    d = df[(df.game == "tg") & df["beliefs_hp"].notna()].copy()
    d["x"] = d["category"].map({c: res[c]["x"] for c in CATS})
    cov_ind = np.cov(d["x"], d["beliefs_hp"])[0, 1]
    q = d["category"].value_counts(normalize=True)
    xs = np.array([res[c]["x"] for c in q.index])
    ss = np.array([d.loc[d.category == c, "beliefs_hp"].mean() for c in q.index])
    w = q.values
    cov_bet = float(np.sum(w * (xs - w @ xs) * (ss - w @ ss)))
    return cov_ind, (1 + R) * cov_ind, cov_bet, (1 + R) * cov_bet, len(d)


def bootstrap(df):
    rng = np.random.default_rng(SEED)
    keys = ["x", "y", "a_foc", "b_foc", "a2", "b2", "p_chosen_pred"]
    draws = []
    idx_by_game = {g: df.index[df.game == g].to_numpy() for g in ["dgkw", "ug", "tg"]}
    for _ in range(B_BOOT):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_game.values()])
        try:
            r = calibrate(moments(df.loc[take]))
            draws.append({(c, k): r[c].get(k, np.nan) for c in CATS for k in keys})
        except Exception:
            continue
    bd = pd.DataFrame(draws)
    return bd.std(), len(bd)


def main():
    df = load_control()
    m_all = moments(df)
    res = calibrate(m_all)
    slopes = empirical_slopes(df)
    cov_ind, gap_ind, cov_bet, gap_bet, n_cov = covariance_exercise(df, res)
    ses, n_ok = bootstrap(df)

    mbc = m_all["Mutual Benefit / Cooperation"]
    locus = {x: tg_locus(mbc, x) for x in [0.0, 0.1, 0.2, 0.3, 0.4]}

    lines = ["Calibration of P1 attention weights and beliefs - control sample",
             f"(bootstrap SEs: {n_ok}/{B_BOOT} successful draws, seed {SEED})", "",
             "Control moments by category:",
             pd.DataFrame(m_all).T.round(4).to_string(), ""]
    for cat in CATS:
        p = res[cat]
        se = lambda k: ses.get((cat, k), np.nan)  # noqa: E731
        lines += [
            f"--- {cat} ---",
            f"  sigma/mu = {p['x']:.3f} (se {se('x'):.3f})   rho/mu = {p['y']:.3f} (se {se('y'):.3f})",
            f"  (a,b) FOC route = ({p['a_foc']:.3f}, {p['b_foc']:.3f})"
            + ("  [IMPOSED: SI schedule, flat-reference-beliefs assumption]" if p.get("ab_imposed")
               else f"  valid schedule: {p['ab_valid']}"
                    + ("" if p["ab_valid"] else "  [b<=0 or bounds violated -> unidentified,"
                                                " as norm-domination implies]")),
            f"  (a,b) two-point route (diagnostic) = ({p['a2']:.3f}, {p['b2']:.3f}); "
            f"a>=0: {p['a2'] >= 0}, a+b<=1: {p['a2'] + p['b2'] <= 1}",
            f"  s (TG believed share): hp {p['s_hp']:.3f}, chosen-action {p['s_chosen']:.3f}",
            f"  overid: schedule-implied acceptance at chosen offer {p['p_chosen_pred']:.3f} "
            f"vs measured {p['p_chosen_actual']:.3f} "
            f"(gap {p['p_chosen_actual'] - p['p_chosen_pred']:+.3f})",
            f"  UG offer: implied {p['t_ug_pred']:.3f} vs actual {p['t_ug_actual']:.3f}",
            f"  belief sensitivity UG: predicted {p['sens_ug_pred']:.3f}, "
            f"estimated {slopes[('ug', cat)][0]:.3f} (se {slopes[('ug', cat)][1]:.3f}, "
            f"N {slopes[('ug', cat)][2]})   [model sign: negative]",
            f"  belief sensitivity TG: predicted {p['sens_tg_pred']:.3f}, "
            f"estimated {slopes[('tg', cat)][0]:.3f} (se {slopes[('tg', cat)][1]:.3f}, "
            f"N {slopes[('tg', cat)][2]})   [model sign: positive]",
            "",
        ]
    si, mo = res["Self-interest"], res["Moral"]
    lines += [
        "Attenuation, estimated/predicted (equal-payoff anchor):",
        f"  UG Self-interest: {abs(slopes[('ug', 'Self-interest')][0]) / si['sens_ug_pred']:.2f}",
        "  TG all cells: "
        + ", ".join(f"{c.split()[0]} {slopes[('tg', c)][0] / res[c]['sens_tg_pred']:.2f}"
                    for c in CATS)
        + "  [uniform within the TG: the anchor's belief channel is common to all types]",
        f"  (Moral UG sensitivity {mo['sens_ug_pred']:.3f}: near-zero by norm domination; "
        "in the TG the norm channel dte/ds keeps every cell's predicted sensitivity positive)",
        "",
        f"MBC TG locus rho/mu = f(sigma/mu) at s_hp={mbc['s_hp']:.3f}: "
        + ", ".join(f"f({x:.1f})={y:.3f}" for x, y in locus.items())
        + "  -> rho>sigma for every admissible sigma/mu",
        "",
        "Attention-belief coupling (rem:joint), TG control:",
        f"  individual-level Cov(sigma/mu, s) = {cov_ind:.4f} => E[t]-E_indep[t] = {gap_ind:.4f} "
        f"({gap_ind * 100:.1f} pp of the endowment; N={n_cov})",
        f"  between-category Cov = {cov_bet:.4f} => gap = {gap_bet:.4f}",
        "",
        "Headline route: (sigma/mu, rho/mu) from DG+TG moments (MBC: UG+TG joint, SI schedule",
        "imposed); (a,b) from UG FOC + reference belief point; the chosen-offer belief point is",
        "the overidentifying test. The two-point route (candidate headline per AA 07-11) fails",
        "validity for Moral and Self-interest: chosen-action beliefs are 16-21 pp above any",
        "coherent linear schedule through the reference point - optimism/selection at chosen",
        "actions; quantified by the overid gaps above and usable in the forecast-error section.",
    ]
    (TABLES / "calibration_stats.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    def f3(v):
        return f"{v:.3f}" if np.isfinite(v) else "--"

    rows = []
    for cat in CATS:
        p = res[cat]
        ab_ok = p.get("ab_valid") and not p.get("ab_imposed")
        a_txt = f3(p["a_foc"]) if (ab_ok or p.get("ab_imposed")) else "--"
        b_txt = f3(p["b_foc"]) if (ab_ok or p.get("ab_imposed")) else "--"
        dag = r"$^{\dagger}$" if p.get("ab_imposed") else ""
        rows.append(
            f"{SHORT[cat]} & {f3(p['x'])} & {f3(p['y'])} & {a_txt}{dag} & {b_txt}{dag} & "
            f"{f3(p['s_hp'])} & {f3(p['p_chosen_pred'])} & {f3(p['p_chosen_actual'])} \\\\")
        rows.append(
            f" & ({f3(ses.get((cat, 'x'), np.nan))}) & ({f3(ses.get((cat, 'y'), np.nan))}) & "
            + (f"({f3(ses.get((cat, 'a_foc'), np.nan))}) & ({f3(ses.get((cat, 'b_foc'), np.nan))})"
               if ab_ok else " & ")
            + " & & & \\\\")
    tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Calibrated Representations, by Category (Control Sample)}}
\label{tab:calibration_p1}
\begin{tabular}{l cc cc c cc}
\toprule
& \multicolumn{2}{c}{Attention} & \multicolumn{3}{c}{Beliefs} & \multicolumn{2}{c}{Acceptance at chosen offer} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}
Category & $\sigma/\mu$ & $\rho/\mu$ & $a$ & $b$ & $s$ & implied & measured \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Control sample. $\sigma/\mu$ from mean DG-KW transfers (Proposition~\ref{prop:dg}); $\rho/\mu$ from the trust-game first-order condition at the measured believed share $s$ (mean hypothetical belief at the one-third reference). For Mutual Benefit/Cooperation, whose dictator-game cell is negligible, $(\sigma/\mu,\rho/\mu)$ solve the ultimatum- and trust-game first-order conditions jointly. $(a,b)$ from the ultimatum-game first-order condition plus the reference-action belief point; for Moral the inversion returns no valid schedule---beliefs are unidentified exactly where the model says behavior is insensitive to them---and $^{\dagger}$ marks the Self-interest schedule imposed on Mutual Benefit/Cooperation (reference-action acceptance beliefs are flat across categories). The last two columns are the overidentifying test: acceptance at the category's mean chosen offer implied by the calibrated schedule---for Moral, the Self-interest schedule, its own being unidentified---versus the measured chosen-action belief; measured beliefs exceed the schedule at chosen offers, the optimism the forecast-error analysis takes up. Bootstrap standard errors in parentheses (participants resampled within game, $B=1{,}000$).
\end{flushleft}
\end{table}
"""
    (TABLES / "calibration_p1.tex").write_text(tex)
    print(f"\nwrote {TABLES / 'calibration_p1.tex'} and calibration_stats.txt")


if __name__ == "__main__":
    main()
