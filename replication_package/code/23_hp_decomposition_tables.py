#!/usr/bin/env python3
"""Generate the two Appendix F decomposition tables (Tables
tab:decomposition_sym_kw_lt and tab:decomposition_sym_pooled_m0m1) from
data/hpmin_sp_moral_all.xlsx.

History: the published versions of these two tables were computed by AA over
all hypothetical-allocation texts rather than the retained-text (hpmin)
sample used everywhere else in Appendix F -- a construction he identified as
an error on recovering his code (2026-07-21) and whose rep/beh splits match
no standard decomposition even on the per-level data (see
verification/aa_perlevel_checks.py). Per AA's own recommendation the tables
are computed here on the hpmin sample, with the paper's canonical
symmetrized (Shapley/Oaxaca--Blinder) construction -- the same object as
Section 5.1's preregistered decomposition (11_oaxaca.py) -- over the five
social-proximity levels, matching the table captions. The table notes report
the moral-7-cell alternative for the mean-allocation row.

Outputs:
  output/tables/decomposition_sym_kw_lt.tex
  output/tables/decomposition_sym_pooled_m0m1.tex
  output/tables/hp_decomposition_stats.txt
Verification: verification/verify_hp_moral_tables.py recomputes both tables
independently and parses these .tex files against its own values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
TABLES = PKG / "output" / "tables"

OUTCOMES = ["Mean Allocation", "Allocation = 4", "Allocation = 6",
            "Allocation = 8", "Allocation $>8$", "Allocation = 12"]

L: list[str] = []


def log(*s: object) -> None:
    L.append(" ".join(str(x) for x in s))
    print(L[-1])


def outcome(d: pd.DataFrame, name: str) -> pd.Series:
    return {"Mean Allocation": d.allocation.astype(float),
            "Allocation = 4": (d.allocation == 4).astype(float),
            "Allocation = 6": (d.allocation == 6).astype(float),
            "Allocation = 8": (d.allocation == 8).astype(float),
            "Allocation $>8$": (d.allocation > 8).astype(float),
            "Allocation = 12": (d.allocation == 12).astype(float)}[name]


def sym_decompose(d: pd.DataFrame, cellcol: str, groupcol: str, g_hi, g_lo,
                  ycol: str) -> tuple[float, float, float]:
    """Symmetrized (Shapley) two-way decomposition of E[y|hi]-E[y|lo]:
    rep = sum_c (q_hi_c - q_lo_c) * mean(y_hi_c, y_lo_c),
    beh = sum_c (q_hi_c + q_lo_c)/2 * (y_hi_c - y_lo_c)."""
    hi, lo = d[d[groupcol] == g_hi], d[d[groupcol] == g_lo]
    qH = hi[cellcol].value_counts(normalize=True)
    qL = lo[cellcol].value_counts(normalize=True)
    yH = hi.groupby(cellcol)[ycol].mean()
    yL = lo.groupby(cellcol)[ycol].mean()
    cells = set(qH.index) | set(qL.index)
    missing = [c for c in cells if c not in qH.index or c not in qL.index]
    if missing:  # the empty-cell rule must never bind here -- fail loudly
        raise ValueError(f"cell(s) empty in one condition: {missing}")
    diff = hi[ycol].mean() - lo[ycol].mean()
    rep = sum((qH[c] - qL[c]) * np.mean([yH[c], yL[c]]) for c in cells)
    beh = sum((qH[c] + qL[c]) / 2 * (yH[c] - yL[c]) for c in cells)
    return diff, rep, beh


def fmt(x: float) -> str:
    """3-dp display with true round-half-up (away from zero). The float
    computation can land a hair below an exact half boundary (e.g. the KW-LT
    mean-allocation diff is exactly 199.5/200 = 0.9975, computed as
    0.99749999...); the epsilon is far below the smallest exact distance any
    of these rationals can sit from a boundary, verified against an
    exact-Fraction recomputation of all 36 displayed values."""
    eps = 1e-9 if x >= 0 else -1e-9
    s = f"{float(x) + eps:.3f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def build_table(d: pd.DataFrame, groupcol: str, g_hi, g_lo, header: str,
                caption: str, label: str, note: str, fname: str) -> None:
    rows = []
    log(f"\n{label} ({header}):")
    for name in OUTCOMES:
        dd = d.assign(_y=outcome(d, name))
        diff, rep, beh = sym_decompose(dd, "social_proximity", groupcol,
                                       g_hi, g_lo, "_y")
        _, repM, behM = sym_decompose(dd, "moral", groupcol, g_hi, g_lo, "_y")
        rows.append((name, diff, rep, beh))
        share = rep / diff
        log(f"  {name:18s} diff {diff:+.4f} SP-5 rep/beh {rep:+.4f}/{beh:+.4f}"
            f" (rep share {share:+.1%}); moral-7 {repM:+.4f}/{behM:+.4f}")
        if name == "Mean Allocation":
            note = note.format(repM=fmt(repM), behM=fmt(behM))
    body = "\n".join(
        f"{name:18s} & {fmt(diff)} & {fmt(rep)} & {fmt(beh)} \\\\"
        for name, diff, rep, beh in rows)
    tex = f"""\\begin{{table}}[!htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Variable}} & \\textbf{{Observed Diff}} & \\textbf{{Representation Component}} & \\textbf{{Behavior Component}} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{{header}}}}} \\\\
{body}
\\bottomrule
\\end{{tabular}}
\\begin{{flushleft}}
\\footnotesize {note}
\\end{{flushleft}}
\\end{{table}}
"""
    out = TABLES / fname
    out.write_text(tex)
    log(f"  wrote {out.relative_to(PKG)}")


def main() -> None:
    d = pd.read_excel(DATA / "hpmin_sp_moral_all.xlsx")
    assert len(d) == 1200 and d.social_proximity.notna().all() \
        and d.moral.notna().all()
    ctl = d[d.Market == 0]

    build_table(
        ctl, "treatment", "kw", "lt", "KW $-$ LT",
        caption=("Symmetrized (Shapley/Oaxaca--Blinder) decomposition of mean "
                 "differences between DG-KW and DG-LT into a representation "
                 "component (differences in the distribution of social "
                 "proximity) and a behavior component (differences in "
                 "allocations conditional on social proximity), Control "
                 "condition, hypothetical-allocation sample. The decomposition "
                 "is a descriptive accounting; no sampling uncertainty is "
                 "attached."),
        label="tab:decomposition_sym_kw_lt",
        note=("Notes: The construction is the symmetrized decomposition of "
              "Section~\\ref{{sec:heterogeneity}}, with representation cells "
              "given by the five social-proximity levels of "
              "Table~\\ref{{tab:social_proximity_hpmin_kw_lt}} ($N=200$ per "
              "game); no cell is empty in either game. Cells defined by the "
              "seven moral categories instead split the mean-allocation gap "
              "into {repM} (representation) and {behM} (behavior)."),
        fname="decomposition_sym_kw_lt.tex")

    build_table(
        d, "Market", 0, 1, "Control $-$ Market",
        caption=("Symmetrized (Shapley/Oaxaca--Blinder) decomposition of mean "
                 "differences between Control and Market into a representation "
                 "component (differences in the distribution of social "
                 "proximity) and a behavior component (differences in "
                 "allocations conditional on social proximity), dictator games "
                 "pooled, hypothetical-allocation sample. The decomposition is "
                 "a descriptive accounting; no sampling uncertainty is "
                 "attached."),
        label="tab:decomposition_sym_pooled_m0m1",
        note=("Notes: The construction is the symmetrized decomposition of "
              "Section~\\ref{{sec:heterogeneity}}, with representation cells "
              "given by the five social-proximity levels of "
              "Table~\\ref{{tab:freqs_market_control_sp}} ($N=400$ Control, "
              "$N=800$ Market); no cell is empty in either condition. Cells "
              "defined by the seven moral categories instead split the "
              "mean-allocation gap into {repM} (representation) and {behM} "
              "(behavior)."),
        fname="decomposition_sym_pooled_m0m1.tex")

    (TABLES / "hp_decomposition_stats.txt").write_text("\n".join(L) + "\n")
    print(f"wrote {TABLES / 'hp_decomposition_stats.txt'}")


if __name__ == "__main__":
    main()
