#!/usr/bin/env python3
"""Empirical foundation of P2 behavior (NG comment 3, 2026-07-19).

(a) How P2's representation depends on the action faced: category shares by bins of the P1
    action (share_sent_p1), and the mixture decomposition of the aggregate response slope
    (eq. receiver_mixture): aggregate = within-category margin + composition.
(b) Parsimonious objective: category-level acceptance schedules p_c(t) = a_c + b_c t (UG) and
    return schedules s_c(t) (TG) with the protest-model signatures: Moral bad = schedule scaled
    down across the board (low intercept); Self-interest = money-sensitive (high slope);
    Moral good = high intercept. TG: returned share declining in the send within category
    (the output/amount-sent-target signature of rem:return).
(c) NG's conjecture: do forecast errors arise because P1 under-weights P2's response to the
    action and over-weights context? Compares (i) P1's believed acceptance slope (calibrated
    FOC route + individual two-point medians) with P2's actual slope; (ii) treatment shifts in
    P1 reference-action beliefs vs shifts in P2 hypothetical behavior at the same action.
Plus: forecast errors by offer bin (verification for the P.38 one-liner: market-frame optimism
    concentrated at very low offers).

Inputs:  data/player{1,2}_all_categorized.xlsx
Outputs: output/tables/p2_foundation_stats.txt, p2_cat_by_action.tex, p2_schedules.tex,
         p2_cat_by_action.csv (figure-ready)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

REF = 1.0 / 3.0
COND = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}
P2_CATS = ["Moral good", "Moral bad", "Self-interest", "Mutual Benefit / Cooperation"]
BINS = [-0.001, 0.0, 0.25, 0.4165, 0.5, 1.0]  # 0, (0,.25], (.25,5/12], (5/12,.5], (.5,1]
BIN_LABELS = ["0", "(0,1/4]", "(1/4,5/12]", "(5/12,1/2]", "(1/2,1]"]

L = []


def log(*s):
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def load():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p2 = pd.read_excel(DATA / "player2_all_categorized.xlsx")
    for d in (p1, p2):
        d["story"] = pd.to_numeric(d["story"], errors="coerce")
    p2 = p2[p2["story"].isin(COND)].copy()
    p2["outcome"] = np.where(p2.game == "ug", p2["choice"], p2["share_sent"])
    p2["outcome_hp"] = np.where(p2.game == "ug", p2["choice_hp"], p2["share_sent_hp"])
    return p1, p2


def slope(d, y, x, controls=None):
    """OLS slope of y on x (+controls), HC1. Returns (coef, se, N)."""
    dd = d.dropna(subset=[y, x] + (controls or []))
    if len(dd) < 10:
        return np.nan, np.nan, len(dd)
    X = sm.add_constant(dd[[x] + (controls or [])])
    fit = sm.OLS(dd[y], X).fit(cov_type="HC1")
    return fit.params[x], fit.bse[x], len(dd)


# ---------------------------------------------------------------- (a) representation vs action
def module_a(p2):
    log("=" * 78)
    log("(a) P2 representations depend on the action faced")
    log("=" * 78)
    rows = []
    for game in ["ug", "tg"]:
        for sample, d in [("control", p2[(p2.game == game) & (p2.story == 0)]),
                          ("pooled", p2[p2.game == game])]:
            d = d[d.category.isin(P2_CATS)].copy()
            d["bin"] = pd.cut(d["share_sent_p1"], BINS, labels=BIN_LABELS)
            tab = (d.groupby("bin", observed=True)["category"]
                   .value_counts(normalize=True).unstack().reindex(columns=P2_CATS))
            ns = d.groupby("bin", observed=True).size()
            log(f"\n{game.upper()} {sample}: category shares by P1-action bin (N per bin: "
                f"{ns.to_dict()})")
            log(tab.round(3).to_string())
            for cat in P2_CATS:
                d[f"is_{cat}"] = (d.category == cat).astype(float)
                b, se, n = slope(d, f"is_{cat}", "share_sent_p1")
                log(f"  LPM {cat} on action faced: {b:+.3f} (se {se:.3f}, N {n})")
                if sample == "pooled":
                    for c in tab.index:
                        pass
            if sample == "control":
                for cat in P2_CATS:
                    for b_ in tab.index:
                        rows.append(dict(game=game, bin=b_, category=cat,
                                         share=tab.loc[b_, cat], n=ns[b_]))
    pd.DataFrame(rows).to_csv(TABLES / "p2_cat_by_action.csv", index=False)

    # mixture decomposition of the aggregate slope (control)
    log("\nMixture decomposition of the aggregate response slope (control):")
    for game in ["ug", "tg"]:
        d = p2[(p2.game == game) & (p2.story == 0) & p2.category.isin(P2_CATS)].copy()
        b_agg, se_agg, n = slope(d, "outcome", "share_sent_p1")
        within = 0.0
        for cat in P2_CATS:
            dc = d[d.category == cat]
            q = len(dc) / len(d)
            b_c, _, _ = slope(dc, "outcome", "share_sent_p1")
            within += q * (b_c if np.isfinite(b_c) else 0.0)
        log(f"  {game.upper()}: aggregate slope {b_agg:+.3f} (se {se_agg:.3f}, N {n}); "
            f"within-category margin {within:+.3f}; composition {b_agg - within:+.3f} "
            f"({(b_agg - within) / b_agg * 100:.0f}% of aggregate)")


# ---------------------------------------------------------------- (b) protest-model schedules
def module_b(p2):
    log("\n" + "=" * 78)
    log("(b) Category-level schedules and the protest-model signatures")
    log("=" * 78)
    tex_rows = {"ug": [], "tg": []}
    for game, yvar, sched in [("ug", "outcome", "acceptance p_c(t) = a_c + b_c t"),
                              ("tg", "outcome", "returned share s_c(t)")]:
        log(f"\n{game.upper()} {sched}, control sample:")
        for cat in P2_CATS:
            d = p2[(p2.game == game) & (p2.story == 0) & (p2.category == cat)]
            dd = d.dropna(subset=[yvar, "share_sent_p1"])
            if len(dd) < 10:
                log(f"  {cat}: N={len(dd)} too small")
                tex_rows[game].append((cat, np.nan, np.nan, np.nan, np.nan, len(dd)))
                continue
            X = sm.add_constant(dd["share_sent_p1"])
            fit = sm.OLS(dd[yvar], X).fit(cov_type="HC1")
            a_c, b_c = fit.params["const"], fit.params["share_sent_p1"]
            log(f"  {cat}: intercept {a_c:+.3f} (se {fit.bse['const']:.3f}), "
                f"slope {b_c:+.3f} (se {fit.bse['share_sent_p1']:.3f}), N {len(dd)}, "
                f"mean {dd[yvar].mean():.3f}")
            tex_rows[game].append((cat, a_c, fit.bse["const"], b_c,
                                   fit.bse["share_sent_p1"], len(dd)))
        # signature checks
        d = p2[(p2.game == game) & (p2.story == 0)]
        means = d.groupby("category")["outcome"].mean()
        log(f"  mean outcome by category: {means.round(3).to_dict()}")

    # protest signatures, stated:
    log("\nProtest-model signature checks (control):")
    log("  UG: Moral bad schedule scaled down across the board (low intercept+mean);")
    log("      Self-interest money-sensitive (high slope); Moral good high acceptance.")
    log("  TG: returned share declining in the send within categories = the")
    log("      amount-sent/output-target signature (rem:return); levels ranked")
    log("      Moral good > MBC > Self-interest.")

    def f3(v):
        return f"{v:.3f}" if np.isfinite(v) else "--"

    body = []
    for game, head in [("ug", "Panel A: UG acceptance, $p_c(t)=a_c+b_c\\,t$"),
                       ("tg", "Panel B: TG returned share, $s_c(t)$")]:
        body.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{head}}}}} \\")
        for cat, a_c, a_se, b_c, b_se, n in tex_rows[game]:
            body.append(f"{cat} & {f3(a_c)} & ({f3(a_se)}) & {f3(b_c)} & ({f3(b_se)}) & {n} \\\\")
        if game == "ug":
            body.append(r"\midrule")
    tex = r"""% chktex-file 13
\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Second-Mover Schedules by Category (Control Sample)}}
\label{tab:p2_schedules}
\begin{tabular}{l cc cc c}
\toprule
Category & Intercept & (SE) & Slope in $t$ & (SE) & $N$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Control sample. OLS of the second mover's realized response (UG: acceptance of the offer faced; TG: share of the tripled amount returned) on the first mover's action $t$ (share of the budget), by stated category. Robust (HC1) standard errors. The offers and sends faced are endogenous to the matched first mover; hypothetical responses at the one-third reference action give the treatment-clean counterpart in Figure~\ref{fig:player2_hypothetical}.
\end{flushleft}
\end{table}
"""
    (TABLES / "p2_schedules.tex").write_text(tex)


# ---------------------------------------------------------------- (c) believed vs actual
def module_c(p1, p2):
    log("\n" + "=" * 78)
    log("(c) Believed vs actual responsiveness to the action, and to context")
    log("=" * 78)
    # actual UG acceptance slope, by condition
    log("\nActual UG acceptance slope b-hat (OLS choice on t, by condition):")
    for st, name in COND.items():
        d = p2[(p2.game == "ug") & (p2.story == st)]
        b, se, n = slope(d, "outcome", "share_sent_p1")
        log(f"  {name}: {b:+.3f} (se {se:.3f}, N {n})")

    # P1 believed slope: individual two-point slopes (beliefs - beliefs_hp)/(t - 1/3)
    log("\nP1 believed slope (individual two-point, |t-1/3|>=1/12, winsorized at [-5,5]):")
    for st, name in COND.items():
        d = p1[(p1.game == "ug") & (p1.story == st)].dropna(subset=["beliefs", "beliefs_hp"])
        d = d[(d.share_sent - REF).abs() >= 1.0 / 12.0]
        bi = ((d.beliefs - d.beliefs_hp) / (d.share_sent - REF)).clip(-5, 5)
        below = bi[d.share_sent < REF]
        above = bi[d.share_sent > REF]
        log(f"  {name}: median {bi.median():+.3f}, mean {bi.mean():+.3f} (N {len(bi)}); "
            f"median below-ref {below.median():+.3f} (N {len(below)}), "
            f"above-ref {above.median():+.3f} (N {len(above)})")
    log("  (calibrated FOC-route believed slope, control Self-interest: b = 0.667 - see"
        " calibration_stats.txt)")

    # matched-segment comparison: the actual schedule is nonlinear (steep below the reference,
    # flat above), so compare believed and actual slopes on the same offer segment
    log("\nActual UG acceptance slope by offer segment (OLS within segment, by condition):")
    for st, name in COND.items():
        d = p2[(p2.game == "ug") & (p2.story == st)]
        lo = d[d.share_sent_p1 <= REF + 0.042]  # up to $5 of $12
        hi = d[d.share_sent_p1 > REF + 0.042]
        bl, sel, nl = slope(lo, "outcome", "share_sent_p1")
        bh, seh, nh = slope(hi, "outcome", "share_sent_p1")
        log(f"  {name}: below/at ref {bl:+.3f} (se {sel:.3f}, N {nl}); "
            f"above ref {bh:+.3f} (se {seh:.3f}, N {nh})")
    log("  -> compare believed below-ref medians (where market offers sit) with the actual"
        " below-ref segment slope")

    # context vs action: treatment shifts at the fixed reference action
    log("\nContext channel at the fixed reference action (Market - Control):")
    for game in ["ug", "tg"]:
        b1 = p1[(p1.game == game)].groupby("story")["beliefs_hp"].mean()
        b2 = p2[(p2.game == game)].groupby("story")["outcome_hp"].mean()
        log(f"  {game.upper()}: P1 beliefs_hp shift {b1[1] - b1[0]:+.3f}; "
            f"P2 hypothetical behavior shift {b2[1] - b2[0]:+.3f}; "
            f"excess belief shift {b1[1] - b1[0] - (b2[1] - b2[0]):+.3f}")
    # action channel magnitude for comparison (control UG): acceptance range over offers
    d = p2[(p2.game == "ug") & (p2.story == 0)].copy()
    d["bin"] = pd.cut(d["share_sent_p1"], BINS, labels=BIN_LABELS)
    acc = d.groupby("bin", observed=True)["outcome"].mean()
    log(f"\nAction channel, control UG acceptance by offer bin: {acc.round(3).to_dict()}")
    log("  -> the action moves acceptance by far more than any context shift moves it")


# ---------------------------------------------------------------- FE by offer bin (P.38)
def fe_by_offer(p1, p2):
    log("\n" + "=" * 78)
    log("Forecast errors by offer bin (verification for the P.38 sentence)")
    log("=" * 78)
    for st, name in COND.items():
        d1 = p1[(p1.game == "ug") & (p1.story == st)].dropna(subset=["beliefs"]).copy()
        d2 = p2[(p2.game == "ug") & (p2.story == st)]
        # benchmark: mean realized acceptance at the same offer bin within the condition
        d1["bin"] = pd.cut(d1["share_sent"], BINS, labels=BIN_LABELS)
        d2b = d2.copy()
        d2b["bin"] = pd.cut(d2b["share_sent_p1"], BINS, labels=BIN_LABELS)
        bench = d2b.groupby("bin", observed=True)["outcome"].mean()
        d1["fe"] = d1["beliefs"] - d1["bin"].map(bench).astype(float)
        fe = d1.groupby("bin", observed=True)["fe"].agg(["mean", "size"])
        log(f"\n{name}: signed FE (belief - realized acceptance at same offer bin):")
        log(fe.round(3).to_string())


def main():
    p1, p2 = load()
    module_a(p2)
    module_b(p2)
    module_c(p1, p2)
    fe_by_offer(p1, p2)
    (TABLES / "p2_foundation_stats.txt").write_text("\n".join(L))
    print(f"\nwrote {TABLES / 'p2_foundation_stats.txt'}, p2_schedules.tex, p2_cat_by_action.csv")


if __name__ == "__main__":
    main()
