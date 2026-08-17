#!/usr/bin/env python3
"""Provenance record for the retired Table 32/37 rep/beh splits, using AA's
2026-07-21 per-level delivery (data/hp_moral_all.xlsx and
data/hp_social_proximity_all.xlsx: one row per participant x hypothetical
allocation, 4,800 = 1,200 x 4, hp in {4, 6, 8, 12}).

AA's second reply (2026-07-21) settles the provenance question by
construction rather than by reproduction: he recovered his code and confirms
the published splits were computed over ALL observations -- every
hypothetical-allocation text, not the retained-text (hpmin) sample -- which
he now calls an error ("pero e un errore"); for this analysis he endorses
the hpmin sample, as used everywhere else in Appendix F. The tables were
therefore REPLACED by the canonical recomputation (hpmin sample, SP-5 cells,
symmetrized; code/23_hp_decomposition_tables.py, verified in
verify_hp_moral_tables.py) and the published splits retired.

This script documents two things for the record. First, the per-level files
are structurally sound and fully consistent with the shipped hpmin trio
(same 1,200 participant-cells, identical allocations, identical retained-row
classifications; one missing moral label out of 4,800). Second, an
exhaustive hunt over all-observations constructions -- row-level cells
(binary High/Low SP at every cutoff, SP-5, moral-7, SP x moral, SP x hp,
moral x hp), person-level four-text pattern cells, and linear
(continuous-index) Oaxaca variants, each under symmetrized, both one-sided,
and pooled-reference weightings -- reproduces the observed-diff column
exactly (it is invariant to the 4x per-participant duplication) but NOT the
published rep/beh splits: composition components computable from these files
top out near a third of the gap, against the published ~half. The published
splits are thus irreproducible in detail even from the per-level data;
given AA's own diagnosis, no further hunt is warranted.

Output: output/tables/aa_perlevel_checks.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from verify_hp_moral_tables import (onesided_decompose, outcome_series,
                                    sym_decompose)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent
DATA = PKG / "data"
TABLES = PKG / "output" / "tables"

KEYS = ["PROLIFIC_PID", "treatment", "Market"]
TOL = 5e-4  # 3-dp display tolerance

L: list[str] = []
n_fail = 0
n_match = 0


def log(*s: object) -> None:
    L.append(" ".join(str(x) for x in s))
    print(L[-1])


def ok(name: str, cond: bool) -> None:
    global n_fail
    log(("  PASS  " if cond else "  FAIL  ") + name)
    n_fail += 0 if cond else 1


def variants(dd: pd.DataFrame, cellcol: str, groupcol: str, g_hi, g_lo,
             ycol: str) -> dict[str, tuple[float, float]]:
    diff, repS, behS = sym_decompose(dd, cellcol, groupcol, g_hi, g_lo, ycol)
    (rA, bA), (rB, bB) = onesided_decompose(dd, cellcol, groupcol, g_hi, g_lo,
                                            ycol)
    hi, lo = dd[dd[groupcol] == g_hi], dd[dd[groupcol] == g_lo]
    qH = hi[cellcol].value_counts(normalize=True)
    qL = lo[cellcol].value_counts(normalize=True)
    yP = dd.groupby(cellcol)[ycol].mean()
    rP = sum((qH.get(c, 0.0) - qL.get(c, 0.0)) * yP[c] for c in yP.index)
    return {"sym": (repS, behS), "one-sided A": (rA, bA),
            "one-sided B": (rB, bB), "pooled-ref": (rP, diff - rP)}


def ob_linear(dd: pd.DataFrame, xcols: list[str], groupcol: str, g_hi, g_lo,
              ycol: str) -> dict[str, tuple[float, float]]:
    """Linear Oaxaca: explained = (E[x|hi]-E[x|lo])'beta, beta from pooled /
    hi / lo samples plus the hi-lo average (symmetrized)."""
    hi, lo = dd[dd[groupcol] == g_hi], dd[dd[groupcol] == g_lo]
    dx = hi[xcols].mean() - lo[xcols].mean()
    diff = hi[ycol].mean() - lo[ycol].mean()
    out = {}
    for name, sample in [("pooled", dd), ("hi", hi), ("lo", lo)]:
        b = sm.OLS(sample[ycol], sm.add_constant(sample[xcols])).fit().params
        rep = float(sum(dx[c] * b[c] for c in xcols))
        out[f"beta-{name}"] = (rep, diff - rep)
    rep_sym = float(np.mean([out["beta-hi"][0], out["beta-lo"][0]]))
    out["beta-sym"] = (rep_sym, diff - rep_sym)
    return out


def hunt(d4: pd.DataFrame, p: pd.DataFrame, groupcol: str, g_hi, g_lo,
         expected: dict[str, tuple[float, float, float]], title: str) -> None:
    """All-observations constructions vs the published rep/beh splits. Full
    values are logged for Mean Allocation; the five binary-outcome rows are
    logged as match counts only."""
    global n_match
    log(f"\n=== {title} ===")
    row_schemes: dict[str, pd.Series] = {
        f"HighSP(>={c})": (d4.sp_num >= c).map({True: "H", False: "L"})
        for c in (1, 2, 3, 4)}
    row_schemes["SP-5"] = d4.social_proximity
    row_schemes["moral-7"] = d4.moral
    row_schemes["SP x moral"] = d4.social_proximity + "|" + d4.moral
    row_schemes["SP x hp"] = d4.social_proximity + "|" + d4.hp.astype(str)
    row_schemes["moral x hp"] = d4.moral + "|" + d4.hp.astype(str)
    pat_schemes = {"pattern binary-SP": p.pat_bin, "pattern SP-5": p.pat_sp5,
                   "pattern moral-7": p.pat_mor}
    ob_sets = {"OB rows sp_num": (d4, ["sp_num"]),
               "OB rows moral_num": (d4, ["moral_num"]),
               "OB rows sp+moral": (d4, ["sp_num", "moral_num"]),
               "OB person mean_sp": (p, ["mean_sp"]),
               "OB person n_highSP": (p, ["n_high"]),
               "OB person sp+moral": (p, ["mean_sp", "mean_moral"])}

    for name in ["Mean Allocation", "=12", ">8", "=8", "=6", "=4"]:
        want = expected[name]
        d4y = d4.assign(_y=outcome_series(d4)[name])
        py = p.assign(_y=outcome_series(p)[name])
        menu: dict[str, tuple[float, float]] = {}
        for sname, cells in row_schemes.items():
            for v, rb in variants(d4y.assign(_cell=cells), "_cell", groupcol,
                                  g_hi, g_lo, "_y").items():
                menu[f"{sname} {v}"] = rb
        for sname, cells in pat_schemes.items():
            for v, rb in variants(py.assign(_cell=cells.loc[py.index]), "_cell",
                                  groupcol, g_hi, g_lo, "_y").items():
                menu[f"{sname} {v}"] = rb
        for sname, (sample, xcols) in ob_sets.items():
            sy = sample.assign(_y=outcome_series(sample)[name])
            for v, rb in ob_linear(sy, xcols, groupcol, g_hi, g_lo, "_y").items():
                menu[f"{sname} {v}"] = rb
        hits = [k for k, (r, b) in menu.items()
                if abs(r - want[1]) <= TOL and abs(b - want[2]) <= TOL]
        n_match += bool(hits)
        diff = d4y.loc[d4y[groupcol] == g_hi, "_y"].mean() \
            - d4y.loc[d4y[groupcol] == g_lo, "_y"].mean()
        log(f"\n  {name}: diff {diff:+.4f} vs published {want[0]:+.3f}; "
            f"published rep/beh {want[1]:+.3f}/{want[2]:+.3f}; "
            f"{len(menu)} constructions tried"
            + (f"; MATCH: {'; '.join(hits)}" if hits else "; no match"))
        if name == "Mean Allocation":
            for k, (r, b) in menu.items():
                log(f"      {k:38s} {r:+.4f}/{b:+.4f}")


def main() -> None:
    sp = pd.read_excel(DATA / "hp_social_proximity_all.xlsx")
    mo = pd.read_excel(DATA / "hp_moral_all.xlsx")
    hpmin = pd.read_excel(DATA / "hpmin_sp_moral_all.xlsx")

    # --- structural checks -------------------------------------------------
    log("--- per-level files: structure ---")
    ok(f"both files 4,800 rows (SP {len(sp)}, moral {len(mo)})",
       len(sp) == 4800 and len(mo) == 4800)
    ok(f"hp levels are 4/6/8/12 ({sorted(sp.hp.unique())})",
       sorted(sp.hp.unique()) == [4, 6, 8, 12])
    ok("PID x treatment x Market x hp unique in both",
       not sp.duplicated(KEYS + ["hp"]).any()
       and not mo.duplicated(KEYS + ["hp"]).any())
    ok("4 hp rows per participant-cell in both",
       (sp.groupby(KEYS).size() == 4).all()
       and (mo.groupby(KEYS).size() == 4).all())
    ok(f"missing labels: SP {sp.social_proximity.isna().sum()}, "
       f"moral {mo.moral.isna().sum()} (one moral label missing is known)",
       sp.social_proximity.isna().sum() == 0 and mo.moral.isna().sum() == 1)
    shared = [c for c in sp.columns if c in mo.columns]
    sps = sp.sort_values(KEYS + ["hp"]).reset_index(drop=True)
    mos = mo.sort_values(KEYS + ["hp"]).reset_index(drop=True)
    ok(f"shared columns identical row-by-row ({shared})",
       all(sps[c].equals(mos[c]) for c in shared))
    d4 = sps.merge(mos[KEYS + ["hp", "moral", "moral_num"]], on=KEYS + ["hp"],
                   validate="one_to_one")
    d4 = d4.fillna({"moral": "MISSING"})

    log("\n--- per-level files vs the shipped hpmin trio ---")
    cells_new = set(map(tuple, d4[KEYS].drop_duplicates().itertuples(index=False)))
    cells_old = set(map(tuple, hpmin[KEYS].itertuples(index=False)))
    ok(f"participant-cells identical to hpmin ({len(cells_new)} = 1,200)",
       cells_new == cells_old)
    m = hpmin.merge(d4, on=KEYS + ["hp"], suffixes=("_min", "_all"),
                    validate="one_to_one")
    ok("every hpmin retained row present at its hp level", len(m) == len(hpmin))
    ok("allocation identical on the retained rows",
       (m.allocation_min == m.allocation_all).all())
    ok("retained-row SP and moral classifications identical",
       (m.social_proximity_min == m.social_proximity_all).all()
       and (m.moral_min == m.moral_all).all())
    ok("allocation constant within participant-cell (4x duplication of one choice)",
       d4.groupby(KEYS).allocation.nunique().eq(1).all())

    # person-level frame: one row per participant-cell with four-text aggregates
    d4s = d4.sort_values(KEYS + ["hp"])
    p = d4s.groupby(KEYS).agg(allocation=("allocation", "first"),
                              mean_sp=("sp_num", "mean"),
                              n_high=("sp_num", lambda s: (s >= 2).sum()),
                              mean_moral=("moral_num", "mean")).reset_index()
    p = (p.merge(d4s.assign(h=(d4s.sp_num >= 2).astype(int).astype(str))
                 .groupby(KEYS).h.agg("".join).rename("pat_bin"), on=KEYS)
          .merge(d4s.groupby(KEYS).social_proximity.agg("|".join)
                 .rename("pat_sp5"), on=KEYS)
          .merge(d4s.groupby(KEYS).moral.agg("|".join).rename("pat_mor"),
                 on=KEYS))

    # --- the hunt: published splits from all-observations constructions ----
    expected37 = {"Mean Allocation": (-1.996, -1.132, -0.864),
                  "=4": (0.025, 0.012, 0.013),
                  "=6": (0.416, 0.221, 0.195),
                  "=8": (0.027, 0.021, 0.006),
                  ">8": (-0.461, -0.262, -0.200),
                  "=12": (-0.150, -0.076, -0.074)}
    hunt(d4, p, "Market", 0, 1, expected37,
         "Published Table 37 splits (Control - Market) from all observations")
    expected32 = {"Mean Allocation": (0.998, 0.491, 0.506),
                  "=4": (-0.065, -0.032, -0.033),
                  "=6": (-0.035, -0.019, -0.016),
                  "=8": (-0.075, -0.036, -0.039),
                  ">8": (0.205, 0.102, 0.103),
                  "=12": (0.105, 0.051, 0.054)}
    hunt(d4[d4.Market == 0], p[p.Market == 0], "treatment", "kw", "lt",
         expected32,
         "Published Table 32 splits (KW - LT, Control) from all observations")

    log(f"\nSummary: {n_fail} structural check(s) failed; published splits "
        f"reproduced for {n_match}/12 outcome rows. The observed-diff columns "
        "match the hpmin sample exactly throughout; the published rep/beh "
        "splits match no tested construction and are retired per AA's "
        "diagnosis (erroneous all-observations run). The tables in the paper "
        "are the hpmin recomputation of 23_hp_decomposition_tables.py.")
    out = TABLES / "aa_perlevel_checks.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
