#!/usr/bin/env python3
"""Category x belief interactions in the accounting exercise (NG comment 4, 2026-07-19).

The model implies category-specific belief sensitivities (sigma/(2b sigma+mu), sigma(1+r)/mu),
so the additive benchmark spec is, by the model's own lights, misspecified. This script re-runs
the P1-action accounting with the interaction:

  spec A (additive):    action ~ categories + beliefs_hp          (current benchmark)
  spec B (interacted):  action ~ categories * beliefs_hp

For each pre-registered comparison (Aid vs Bonus, Market vs Control) and game, the predicted
treatment effect is beta'(xbar_T - xbar_C) on the pooled-fit coefficients - with spec B this
includes the movement of the category-belief cross-moments, the object NG asks about. DG-KW has
no belief elicitation: categories only (specs coincide); included so the 6-cell fit comparison
matches the accounting's P1-action block.

Inputs:  data/player1_all_categorized.xlsx
Outputs: output/tables/interaction_accounting.tex, interaction_accounting_stats.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
COMPARISONS = [("Aid vs Bonus", 4, 2), ("Market vs Control", 1, 0)]
GAMES = [("dgkw", "DG-KW"), ("ug", "UG"), ("tg", "TG")]

L = []


def log(*s):
    line = " ".join(str(x) for x in s)
    L.append(line)
    print(line)


def load():
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p1 = p1[p1.category.isin(CATS)].copy()
    p1["cat"] = pd.Categorical(p1.category, categories=CATS)
    return p1


def predict_te(fit, d, treated, baseline):
    """beta'(xbar_T - xbar_C) via model-matrix means."""
    from patsy import dmatrix
    X = dmatrix(fit.model.data.design_info, d, return_type="dataframe")
    xt = X[d.story.values == treated].mean()
    xc = X[d.story.values == baseline].mean()
    return float(fit.params @ (xt - xc))


def main():
    p1 = load()
    results = []
    for comp, treated, baseline in COMPARISONS:
        for game, glabel in GAMES:
            d = p1[(p1.game == game) & p1.story.isin([treated, baseline])].copy()
            has_beliefs = d["beliefs_hp"].notna().sum() > 100
            if has_beliefs:
                d = d.dropna(subset=["beliefs_hp"])
                fA = smf.ols("share_sent ~ cat + beliefs_hp", data=d).fit(cov_type="HC1")
                fB = smf.ols("share_sent ~ cat * beliefs_hp", data=d).fit(cov_type="HC1")
            else:
                fA = smf.ols("share_sent ~ cat", data=d).fit(cov_type="HC1")
                fB = fA
            actual = (d.loc[d.story == treated, "share_sent"].mean()
                      - d.loc[d.story == baseline, "share_sent"].mean())
            pA = predict_te(fA, d, treated, baseline)
            pB = predict_te(fB, d, treated, baseline)
            results.append(dict(comparison=comp, game=glabel, actual=actual,
                                pred_additive=pA, pred_interacted=pB,
                                n=len(d), r2_add=fA.rsquared, r2_int=fB.rsquared,
                                beliefs=has_beliefs))
            log(f"{comp:>18} {glabel:>6}: actual {actual:+.3f} | additive {pA:+.3f} "
                f"(ratio {pA / actual:+.2f}) | interacted {pB:+.3f} (ratio {pB / actual:+.2f})"
                f" | N {len(d)}" + ("" if has_beliefs else "  [categories only - no beliefs]"))
            if has_beliefs:
                ints = {k: v for k, v in fB.params.items() if ":" in k}
                log("      interaction terms: "
                    + ", ".join(f"{k.split('T.')[-1].rstrip(']')}x belief {v:+.3f}"
                                f" (p {fB.pvalues[k]:.3f})" for k, v in ints.items()))

    res = pd.DataFrame(results)
    log("\nFit across the 6 P1-action cells (actual on predicted, OLS):")
    for spec, col in [("additive", "pred_additive"), ("interacted", "pred_interacted")]:
        import statsmodels.api as sm
        f = sm.OLS(res.actual, sm.add_constant(res[col])).fit()
        log(f"  {spec}: slope {f.params.iloc[1]:.3f}, R2 {f.rsquared:.3f}; "
            f"mean |pred|/|actual| on the four largest cells "
            f"{(res.reindex(res.actual.abs().sort_values(ascending=False).index[:4])[col].abs() / res.reindex(res.actual.abs().sort_values(ascending=False).index[:4]).actual.abs()).mean():.2f}")

    def f3(v):
        return f"{v:+.3f}"

    rows = []
    for _, r in res.iterrows():
        rows.append(f"{r.comparison} & {r.game} & {f3(r.actual)} & {f3(r.pred_additive)} & "
                    f"{f3(r.pred_interacted)} & {r.n} \\\\")
    tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Predicted Treatment Effects on Actions: Additive versus Interacted Specification}}
\label{tab:interaction_accounting}
\begin{tabular}{ll ccc c}
\toprule
Comparison & Game & Actual & \multicolumn{2}{c}{Predicted} & $N$ \\
\cmidrule(lr){4-5}
& & & Additive & Interacted & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Player 1 actions, classified sample with non-missing hypothetical beliefs (UG, TG). Additive: action on category dummies and the hypothetical belief (the benchmark specification); Interacted: action on categories, the belief, and their interactions, so predicted effects include the treatment-induced movement of the category-belief cross-moments. Predicted effects are $\hat\beta'(\bar{x}_T-\bar{x}_C)$ on the pooled two-condition fit. DG-KW has no belief elicitation, so both specifications reduce to categories only.
\end{flushleft}
\end{table}
"""
    (TABLES / "interaction_accounting.tex").write_text(tex)
    (TABLES / "interaction_accounting_stats.txt").write_text("\n".join(L))
    print(f"\nwrote {TABLES / 'interaction_accounting.tex'} and interaction_accounting_stats.txt")


if __name__ == "__main__":
    main()
