#!/usr/bin/env python3
"""Verify ALL Appendix F exhibits from data/hpmin_sp_moral_all.xlsx
(AA's 2026-07-21 delivery: the SP file plus the old 7-category moral-scheme
classification, column `moral`). This closes the replication-package gap
flagged by verify_hp_appendix_tables.py: Tables 33/34/36 were computed over
the moral cells, which until now existed only in AA's original workbook.

Covered (hardcoded in main.tex except 32/37, which are generated):
  Table 30 tab:social_proximity_hpmin_kw_lt   SP shares + mean allocation, Control
  Table 31 tab:highsp_regressions_hpmin       High-SP OLS, Control
  Table 32 tab:decomposition_sym_kw_lt        KW-LT decomposition incl. rep/beh splits
  Table 33 tab:moral_kw_lt                    moral shares + mean allocation, Control
  Table 34 tab:moral_regressions              moral-category OLS, Control
  Table 35 tab:freqs_market_control_sp        SP shares, Market vs Control
  Table 36 tab:freqs_market_control_moral     moral shares, Market vs Control + p-values
  Table 37 tab:decomposition_sym_pooled_m0m1  Control-Market decomposition incl. splits
  Figs 19/20 fig:hp_sp_moral_{ctrl,mkt}       the 35 annotated cells of each heatmap
Plus: row-by-row consistency of the new file against
data/hpmin_social_proximity_all.xlsx, and regenerated heatmaps
(output/figures/hp_sp_moral_corr_{ctrl,mkt}_repro.png) for visual comparison.

Tables 32/37 are GENERATED tables since 2026-07-21: AA recovered his code
and confirmed the published rep/beh splits came from an erroneous
all-observations run (every hypothetical-allocation text instead of the
retained-text sample; irreproducible in detail even from the per-level
files, see aa_perlevel_checks.py), and endorsed recomputing on the hpmin
sample used everywhere else in this appendix. The tables in the paper are
now emitted by code/23_hp_decomposition_tables.py (hpmin sample, SP-5
cells, symmetrized); this script recomputes both decompositions
independently and grades the expected display values AND the .tex files on
disk, including the moral-7 alternative quoted in each table's notes.
The p-values of Tables 35/36 are the two-sided test of equal proportions
WITH continuity correction (R prop.test default = Yates-corrected chi2),
identified from AA's 2026-07-21 reply and confirmed by aa_reply_checks.py:
chi2-Yates reproduces AA's recomputed 0.137/0.536/0.0288 at displayed
precision, so the three numeric entries were corrected in main.tex
(0.13/0.42/0.015 -> 0.14/0.54/0.029; the originals matched no single
standard test) and are now graded exactly against chi2-Yates.

Outputs: output/tables/verify_hp_moral_stats.txt
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent
DATA = PKG / "data"
TABLES = PKG / "output" / "tables"
FIGURES = PKG / "output" / "figures"

SP_LEVELS = ["No mention of recipient", "Abstract stranger", "Anonymous peer",
             "Teammate / coworker", "Friend"]
MORAL_FIG_ORDER = ["Generous", "Entitlement", "Egalitarian", "Greedy",
                   "Theft", "Need", "Neutral"]

L: list[str] = []
n_fail = 0


def log(*s: object) -> None:
    L.append(" ".join(str(x) for x in s))
    print(L[-1])


def ok(name: str, cond: bool) -> None:
    global n_fail
    log(("  PASS  " if cond else "  FAIL  ") + name)
    n_fail += 0 if cond else 1


def close(a: float, b: float, dp: int) -> bool:
    """True when a could round to the dp-digit display value b."""
    return abs(a - b) <= 0.5 * 10 ** (-dp) + 1e-9


def stars(p: float, with_10pct: bool) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if with_10pct and p < 0.10:
        return "*"
    return ""


def sym_decompose(d: pd.DataFrame, cellcol: str, groupcol: str, g_hi, g_lo,
                  ycol: str) -> tuple[float, float, float]:
    """Symmetrized (Shapley) two-way decomposition of E[y|hi]-E[y|lo]."""
    hi, lo = d[d[groupcol] == g_hi], d[d[groupcol] == g_lo]
    qH = hi[cellcol].value_counts(normalize=True)
    qL = lo[cellcol].value_counts(normalize=True)
    yH = hi.groupby(cellcol)[ycol].mean()
    yL = lo.groupby(cellcol)[ycol].mean()
    diff = hi[ycol].mean() - lo[ycol].mean()
    rep = beh = 0.0
    for c in sorted(set(qH.index) | set(qL.index)):
        qh, ql = qH.get(c, 0.0), qL.get(c, 0.0)
        yh, yl = yH.get(c, np.nan), yL.get(c, np.nan)
        rep += (qh - ql) * np.nanmean([yh, yl])
        if np.isfinite(yh) and np.isfinite(yl):
            beh += (qh + ql) / 2 * (yh - yl)
    return diff, rep, beh


OUTCOME_ORDER = ["Mean Allocation", "=12", ">8", "=8", "=6", "=4"]


def outcome_series(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "Mean Allocation": d.allocation.astype(float),
        "=12": (d.allocation == 12).astype(float),
        ">8": (d.allocation > 8).astype(float),
        "=8": (d.allocation == 8).astype(float),
        "=6": (d.allocation == 6).astype(float),
        "=4": (d.allocation == 4).astype(float),
    }


def check_ols_column(y: pd.Series, X: pd.DataFrame, colname: str,
                     expected: dict[str, tuple[float, float, str]],
                     r2: float, with_10pct: bool) -> None:
    """expected: regressor -> (coef, se, stars); classical (nonrobust) SEs."""
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    for reg, (c, s, st) in expected.items():
        key = "const" if reg == "const" else reg
        got_c, got_s = fit.params[key], fit.bse[key]
        got_st = stars(fit.pvalues[key], with_10pct)
        ok(f"{colname} {reg}: coef {got_c:+.4f} se {got_s:.4f} [{got_st or 'ns'}] "
           f"vs hardcoded {c:+.3f} ({s:.3f}) [{st or 'ns'}]",
           close(got_c, c, 3) and close(got_s, s, 3) and got_st == st)
    ok(f"{colname} R2 {fit.rsquared:.4f} vs hardcoded {r2:.3f}",
       close(fit.rsquared, r2, 3))
    ok(f"{colname} N {int(fit.nobs)} vs hardcoded 400", int(fit.nobs) == 400)


def share_pvalue_row(mkt: pd.DataFrame, ctl: pd.DataFrame, col: str, lvl: str,
                     wm: float, wc: float, wp: str) -> None:
    gm = 100 * (mkt[col] == lvl).mean()
    gc = 100 * (ctl[col] == lvl).mean()
    cnt = np.array([(mkt[col] == lvl).sum(), (ctl[col] == lvl).sum()])
    tot = np.array([len(mkt), len(ctl)])
    table = np.array([cnt, tot - cnt]).T
    battery = {
        "chi2": stats.chi2_contingency(table, correction=False).pvalue,
        "chi2-Yates": stats.chi2_contingency(table, correction=True).pvalue,
        "G": stats.chi2_contingency(table, correction=False,
                                    lambda_="log-likelihood").pvalue,
        "G-Yates": stats.chi2_contingency(table, correction=True,
                                          lambda_="log-likelihood").pvalue,
        "Fisher": stats.fisher_exact(table)[1],
        "Barnard": stats.barnard_exact(table).pvalue,
    }
    if wp == "<0.001":
        p_ok, note = battery["chi2-Yates"] < 0.001, "chi2-Yates (prop.test)"
    else:
        want = float(wp)
        dp = len(wp.split(".")[1])
        p_ok = close(battery["chi2-Yates"], want, dp)
        note = (f"chi2-Yates (prop.test, AA's test) {battery['chi2-Yates']:.4f}"
                + ("" if p_ok else " DOES NOT round to hardcoded")
                + "; " + ", ".join(f"{k} {v:.4f}" for k, v in battery.items()))
    ok(f"{lvl}: Market {gm:.4f} / Control {gc:.4f} vs {wm} / {wc}; "
       f"p vs '{wp}' [{note}]",
       close(gm, wm, 2) and close(gc, wc, 2) and p_ok)


def onesided_decompose(d: pd.DataFrame, cellcol: str, groupcol: str, g_hi, g_lo,
                       ycol: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two exact one-sided (Oaxaca) pairs: A = (rep at lo-condition cell
    means, beh at hi-condition shares); B = the reverse weighting."""
    hi, lo = d[d[groupcol] == g_hi], d[d[groupcol] == g_lo]
    qH = hi[cellcol].value_counts(normalize=True)
    qL = lo[cellcol].value_counts(normalize=True)
    yH = hi.groupby(cellcol)[ycol].mean()
    yL = lo.groupby(cellcol)[ycol].mean()
    cells = set(qH.index) | set(qL.index)
    repA = sum((qH.get(c, 0.0) - qL.get(c, 0.0)) * yL.get(c, yH.get(c))
               for c in cells)
    behA = sum(qH.get(c, 0.0) * (yH[c] - yL[c])
               for c in set(qH.index) & set(qL.index))
    repB = sum((qH.get(c, 0.0) - qL.get(c, 0.0)) * yH.get(c, yL.get(c))
               for c in cells)
    behB = sum(qL.get(c, 0.0) * (yH[c] - yL[c])
               for c in set(qH.index) & set(qL.index))
    return (repA, behA), (repB, behB)


TEX_ROW = {"Mean Allocation": "Mean Allocation", "=4": "Allocation = 4",
           "=6": "Allocation = 6", "=8": "Allocation = 8",
           ">8": "Allocation $>8$", "=12": "Allocation = 12"}


def check_decomposition(d: pd.DataFrame, groupcol: str, g_hi, g_lo,
                        expected: dict[str, tuple[float, float, float]],
                        texname: str, title: str) -> None:
    """Verify a generated decomposition table (23_hp_decomposition_tables.py):
    recompute the SP-5 symmetrized decomposition on the hpmin sample, grade
    the expected display values, and parse the .tex on disk against them;
    also recompute the moral-7 alternative quoted in the table notes."""
    log(f"\n--- {title} ---")
    tex = (TABLES / texname).read_text()
    rows = {}
    for line in tex.splitlines():
        for name, rowlab in TEX_ROW.items():
            if line.startswith(rowlab + " "):
                cells = [c.strip().replace("$-$", "-").rstrip("\\").strip()
                         for c in line.split("&")[1:]]
                rows[name] = tuple(float(c) for c in cells)
    ok(f"{texname}: all six outcome rows found", len(rows) == 6)
    for name, y in outcome_series(d).items():
        dd = d.assign(_y=y)
        want = expected[name]
        diff, repS, behS = sym_decompose(dd, "social_proximity", groupcol,
                                         g_hi, g_lo, "_y")
        ok(f"{name}: diff {diff:+.4f} rep {repS:+.4f} beh {behS:+.4f} vs "
           f"expected {want[0]:+.3f}/{want[1]:+.3f}/{want[2]:+.3f}",
           close(diff, want[0], 3) and close(repS, want[1], 3)
           and close(behS, want[2], 3))
        ok(f"{name}: .tex row {rows.get(name)} matches expected",
           rows.get(name) == want)
        if name == "Mean Allocation":
            _, repM, behM = sym_decompose(dd, "moral", groupcol, g_hi, g_lo, "_y")
            m = re.search(r"into (\S+) \(representation\) and (\S+) "
                          r"\(behavior\)", tex.replace("$-$", "-"))
            if m is None:
                ok("notes moral-7 quoted values found in .tex", False)
            else:
                ok(f"notes moral-7 split {repM:+.4f}/{behM:+.4f} vs quoted "
                   f"{m.group(1)}/{m.group(2)}",
                   close(repM, float(m.group(1)), 3)
                   and close(behM, float(m.group(2)), 3))


def check_fig_grid(d: pd.DataFrame, cond: str, expected: list[list[float]],
                   fname: str) -> None:
    log(f"\n--- fig:hp_sp_moral_{cond} ({fname}): 35 annotated cells ---")
    n = len(d)
    grid = np.zeros((len(MORAL_FIG_ORDER), len(SP_LEVELS)))
    bad = []
    for i, m in enumerate(MORAL_FIG_ORDER):
        for j, s in enumerate(SP_LEVELS):
            grid[i, j] = ((d.moral == m) & (d.social_proximity == s)).sum() / n
            if not close(grid[i, j], expected[i][j], 2):
                bad.append(f"{m} x {s}: computed {grid[i, j]:.4f} vs annotated "
                           f"{expected[i][j]:.2f}")
    ok(f"all 35 cells match the PNG annotations at 2dp (N={n})", not bad)
    for b in bad:
        log("      MISMATCH " + b)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_xticks(range(len(SP_LEVELS)), SP_LEVELS, rotation=40, ha="right")
    ax.set_yticks(range(len(MORAL_FIG_ORDER)), MORAL_FIG_ORDER)
    ax.set_xlabel("Social Proximity")
    ax.set_ylabel("Moral Categories")
    title_cond = "Control" if cond == "ctrl" else "Market"
    ax.set_title(f"Relative frequency grid: Moral × SP ({title_cond})")
    fig.colorbar(im, ax=ax, label="Relative frequency")
    fig.tight_layout()
    out = FIGURES / f"hp_sp_moral_corr_{cond}_repro.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    log(f"      regenerated heatmap written to {out.relative_to(PKG)}")


def main() -> None:
    d = pd.read_excel(DATA / "hpmin_sp_moral_all.xlsx")
    sp = pd.read_excel(DATA / "hpmin_social_proximity_all.xlsx")

    # --- file consistency: new file = SP file + moral columns ---
    log("--- data/hpmin_sp_moral_all.xlsx vs data/hpmin_social_proximity_all.xlsx ---")
    ok(f"N={len(d)} rows, both files", len(d) == 1200 and len(sp) == 1200)
    keys = ["PROLIFIC_PID", "treatment", "Market"]
    ok("PROLIFIC_PID x treatment x Market unique in both",
       not d.duplicated(keys).any() and not sp.duplicated(keys).any())
    n_rep = d.PROLIFIC_PID.duplicated().sum()
    log(f"  note: {n_rep} PIDs appear in two cells (repeat participation across "
        "the separately recruited DG cells) — identical in all three hpmin "
        "files, a property of the underlying data, not of this delivery; "
        "PID-only merges against these files fan out")
    shared = [c for c in sp.columns if c in d.columns]
    dm = d.sort_values(keys).reset_index(drop=True)
    sm_ = sp.sort_values(keys).reset_index(drop=True)
    mismatch = [c for c in shared if not dm[c].equals(sm_[c])]
    ok(f"shared columns identical row-by-row ({shared})", not mismatch)
    for c in mismatch:
        log(f"      column '{c}' differs in {(dm[c] != sm_[c]).sum()} rows")
    ok("moral has the 7 categories, no missing",
       set(d.moral.dropna()) == {"Need", "Egalitarian", "Generous", "Neutral",
                                 "Entitlement", "Greedy", "Theft"}
       and d.moral.notna().all())
    cells = d.groupby(["treatment", "Market"]).size()
    log(f"cells (treatment x Market): {cells.to_dict()}")

    ctl = d[d.Market == 0]
    mkt = d[d.Market == 1]
    kw, lt = ctl[ctl.treatment == "kw"], ctl[ctl.treatment == "lt"]

    # --- Table 30: SP shares + mean allocation by game, Control ---
    log("\n--- tab:social_proximity_hpmin_kw_lt (Table 30) ---")
    expected30 = {"No mention of recipient": (26.50, 31.00, 8.74, 7.39),
                  "Abstract stranger": (28.00, 28.50, 9.31, 7.10),
                  "Anonymous peer": (17.50, 19.00, 6.00, 6.24),
                  "Teammate / coworker": (7.50, 5.50, 6.60, 7.09),
                  "Friend": (20.50, 16.00, 6.83, 6.14)}
    for lvl, (sk, sl, mk, ml) in expected30.items():
        gsk = 100 * (kw.social_proximity == lvl).mean()
        gsl = 100 * (lt.social_proximity == lvl).mean()
        gmk = kw.loc[kw.social_proximity == lvl, "allocation"].mean()
        gml = lt.loc[lt.social_proximity == lvl, "allocation"].mean()
        ok(f"{lvl}: shares {gsk:.2f}/{gsl:.2f} alloc {gmk:.3f}/{gml:.3f} "
           f"vs {sk}/{sl} and {mk}/{ml}",
           close(gsk, sk, 2) and close(gsl, sl, 2)
           and close(gmk, mk, 2) and close(gml, ml, 2))

    # --- Table 31: High-SP OLS, Control (stars: ***/** only) ---
    log("\n--- tab:highsp_regressions_hpmin (Table 31) ---")
    X31 = pd.DataFrame({"High SP": (ctl.sp_num >= 2).astype(float)})
    expected31 = {
        "Mean Allocation": ({"High SP": (-1.703, 0.220, "***"),
                             "const": (8.101, 0.144, "***")}, 0.131),
        # R2 0.101: main.tex said 0.102 until 2026-07-21; exact value 0.1015
        # (every other cell of the column matches) -> corrected in the paper
        "=12": ({"High SP": (-0.228, 0.034, "***"),
                 "const": (0.246, 0.022, "***")}, 0.101),
        ">8": ({"High SP": (-0.299, 0.039, "***"),
                "const": (0.351, 0.026, "***")}, 0.126),
        "=8": ({"High SP": (-0.028, 0.034, ""),
                "const": (0.145, 0.022, "***")}, 0.002),
        "=6": ({"High SP": (0.357, 0.047, "***"),
                "const": (0.364, 0.031, "***")}, 0.125),
        "=4": ({"High SP": (0.006, 0.019, ""),
                "const": (0.035, 0.013, "***")}, 0.000),
    }
    ys31 = outcome_series(ctl)
    for name in OUTCOME_ORDER:
        exp, r2 = expected31[name]
        check_ols_column(ys31[name], X31, name, exp, r2, with_10pct=False)

    # --- Table 32: KW - LT decomposition, Control (generated, SP-5 sym) ---
    expected32 = {"Mean Allocation": (0.998, -0.067, 1.064),
                  "=4": (-0.065, 0.002, -0.067),
                  "=6": (-0.035, 0.008, -0.043),
                  "=8": (-0.075, 0.000, -0.075),
                  ">8": (0.205, -0.010, 0.215),
                  "=12": (0.105, -0.011, 0.116)}
    check_decomposition(ctl, "treatment", "kw", "lt", expected32,
                        "decomposition_sym_kw_lt.tex",
                        "tab:decomposition_sym_kw_lt (Table 32): KW - LT, Control")

    # --- Table 33: moral shares + mean allocation by game, Control ---
    log("\n--- tab:moral_kw_lt (Table 33) ---")
    expected33 = {"Need": (2.50, 1.00, 11.40, 6.00),
                  "Egalitarian": (32.00, 27.50, 6.02, 5.91),
                  "Generous": (33.00, 37.50, 7.43, 6.03),
                  "Neutral": (8.50, 11.00, 8.00, 7.48),
                  "Entitlement": (10.50, 11.00, 9.90, 8.48),
                  "Greedy": (13.00, 8.00, 10.96, 8.75),
                  "Theft": (0.50, 4.00, 12.00, 11.75)}
    for lvl, (sk, sl, mk, ml) in expected33.items():
        gsk = 100 * (kw.moral == lvl).mean()
        gsl = 100 * (lt.moral == lvl).mean()
        gmk = kw.loc[kw.moral == lvl, "allocation"].mean()
        gml = lt.loc[lt.moral == lvl, "allocation"].mean()
        ok(f"{lvl}: shares {gsk:.2f}/{gsl:.2f} alloc {gmk:.3f}/{gml:.3f} "
           f"vs {sk}/{sl} and {mk}/{ml}",
           close(gsk, sk, 2) and close(gsl, sl, 2)
           and close(gmk, mk, 2) and close(gml, ml, 2))

    # --- Table 34: moral-category OLS, Control, baseline Neutral (***/**/*) ---
    log("\n--- tab:moral_regressions (Table 34) ---")
    X34 = pd.get_dummies(ctl.moral).astype(float).drop(columns="Neutral")
    expected34 = {
        "Mean Allocation": ({"Generous": (-1.021, 0.315, "***"),
                             "Entitlement": (1.469, 0.385, "***"),
                             "Egalitarian": (-1.739, 0.321, "***"),
                             "Greedy": (2.414, 0.387, "***"),
                             "Theft": (4.073, 0.644, "***"),
                             "Need": (2.152, 0.714, "***"),
                             "const": (7.705, 0.279, "***")}, 0.452),
        "=12": ({"Generous": (-0.144, 0.053, "***"),
                 "Entitlement": (0.146, 0.064, "**"),
                 "Egalitarian": (-0.179, 0.054, "***"),
                 "Greedy": (0.321, 0.065, "***"),
                 "Theft": (0.709, 0.107, "***"),
                 "Need": (0.392, 0.119, "***"),
                 "const": (0.179, 0.046, "***")}, 0.342),
        ">8": ({"Generous": (-0.103, 0.060, "*"),
                "Entitlement": (0.188, 0.073, "**"),
                "Egalitarian": (-0.231, 0.061, "***"),
                "Greedy": (0.484, 0.074, "***"),
                "Theft": (0.769, 0.122, "***"),
                "Need": (0.484, 0.136, "***"),
                "const": (0.231, 0.053, "***")}, 0.377),
        "=8": ({"Generous": (-0.169, 0.058, "***"),
                "Entitlement": (0.113, 0.071, ""),
                "Egalitarian": (-0.248, 0.059, "***"),
                "Greedy": (-0.163, 0.072, "**"),
                "Theft": (-0.282, 0.119, "**"),
                "Need": (-0.282, 0.132, "**"),
                "const": (0.282, 0.051, "***")}, 0.116),
        "=6": ({"Generous": (0.278, 0.073, "***"),
                "Entitlement": (-0.166, 0.089, "*"),
                "Egalitarian": (0.609, 0.074, "***"),
                "Greedy": (-0.187, 0.090, "**"),
                "Theft": (-0.282, 0.149, "*"),
                "Need": (0.004, 0.165, ""),
                "const": (0.282, 0.064, "***")}, 0.362),
        "=4": ({"Generous": (-0.009, 0.034, ""),
                "Entitlement": (-0.051, 0.042, ""),
                "Egalitarian": (0.008, 0.035, ""),
                "Greedy": (-0.051, 0.042, ""),
                "Theft": (-0.051, 0.070, ""),
                "Need": (-0.051, 0.078, ""),
                "const": (0.051, 0.030, "*")}, 0.014),
    }
    ys34 = outcome_series(ctl)
    for name in OUTCOME_ORDER:
        exp, r2 = expected34[name]
        check_ols_column(ys34[name], X34, name, exp, r2, with_10pct=True)

    # --- Table 35: SP shares, Market vs Control ---
    log("\n--- tab:freqs_market_control_sp (Table 35) ---")
    expected35 = {"No mention of recipient": (75.88, 28.75),
                  "Abstract stranger": (10.75, 28.25),
                  "Anonymous peer": (5.00, 18.25),
                  "Teammate / coworker": (2.13, 6.50),
                  "Friend": (6.25, 18.25)}
    for lvl, (wm, wc) in expected35.items():
        share_pvalue_row(mkt, ctl, "social_proximity", lvl, wm, wc, "<0.001")

    # --- Table 36: moral shares, Market vs Control ---
    log("\n--- tab:freqs_market_control_moral (Table 36) ---")
    expected36 = {"Egalitarian": (7.25, 29.75, "<0.001"),
                  "Entitlement": (22.50, 10.75, "<0.001"),
                  "Generous": (11.13, 35.25, "<0.001"),
                  "Greedy": (7.75, 10.50, "0.14"),
                  "Need": (2.50, 1.75, "0.54"),
                  "Neutral": (48.25, 9.75, "<0.001"),
                  "Theft": (0.63, 2.25, "0.029")}
    for lvl, (wm, wc, wp) in expected36.items():
        share_pvalue_row(mkt, ctl, "moral", lvl, wm, wc, wp)

    # --- Table 37: Control - Market decomposition, pooled (generated) ---
    expected37 = {"Mean Allocation": (-1.996, -0.494, -1.502),
                  "=4": (0.025, -0.002, 0.027),
                  "=6": (0.416, 0.097, 0.319),
                  "=8": (0.028, -0.012, 0.040),
                  ">8": (-0.461, -0.083, -0.378),
                  "=12": (-0.150, -0.047, -0.103)}
    check_decomposition(d, "Market", 0, 1, expected37,
                        "decomposition_sym_pooled_m0m1.tex",
                        "tab:decomposition_sym_pooled_m0m1 (Table 37): Control - Market")

    # --- Figures 19/20: annotated relative-frequency grids ---
    fig_ctrl = [[0.09, 0.14, 0.02, 0.02, 0.08],
                [0.05, 0.03, 0.00, 0.01, 0.01],
                [0.02, 0.02, 0.15, 0.03, 0.07],
                [0.04, 0.05, 0.00, 0.00, 0.01],
                [0.01, 0.01, 0.00, 0.00, 0.00],
                [0.01, 0.01, 0.00, 0.00, 0.00],
                [0.07, 0.02, 0.00, 0.00, 0.01]]
    fig_mkt = [[0.05, 0.02, 0.01, 0.01, 0.02],
               [0.17, 0.03, 0.00, 0.01, 0.01],
               [0.01, 0.01, 0.03, 0.01, 0.01],
               [0.05, 0.02, 0.00, 0.00, 0.00],
               [0.00, 0.00, 0.00, 0.00, 0.00],
               [0.02, 0.00, 0.00, 0.00, 0.00],
               [0.45, 0.01, 0.00, 0.00, 0.01]]
    check_fig_grid(ctl, "ctrl", fig_ctrl, "hp_sp_moral_corr_ctrl.png")
    check_fig_grid(mkt, "mkt", fig_mkt, "hp_sp_moral_corr_mkt.png")

    log(f"\nSummary: {'all checks PASS' if n_fail == 0 else f'{n_fail} check(s) FAILED'}. "
        "Tables 32/37 are generated (23_hp_decomposition_tables.py, hpmin "
        "sample, SP-5 symmetrized) and fully verified, closing the last "
        "Appendix F gap; the retired all-observations splits are documented "
        "in aa_perlevel_checks.py.")
    (TABLES / "verify_hp_moral_stats.txt").write_text("\n".join(L) + "\n")
    print(f"wrote {TABLES / 'verify_hp_moral_stats.txt'}")


if __name__ == "__main__":
    main()
