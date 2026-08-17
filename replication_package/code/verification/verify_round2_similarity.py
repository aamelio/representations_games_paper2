#!/usr/bin/env python3
"""Independent from-scratch recomputation of Table 6 (tab:similarity_categories)
and Figure 6 (fig:similarity_h5), 2026-07-20 (pre-call verification for NG).

Deliberately does NOT import or mirror llm_similarity/round2/04_analysis.py:
every aggregate is rebuilt directly from the raw recordings
(recording_round2.csv, recording_splits_round2.csv) and compared against
(a) the published table file output/tables/similarity_categories.tex (parsed),
(b) the published Figure-6 statistics quoted in main.tex (slope 0.0122,
    se 0.0025, R^2 0.59, Spearman 0.57),
(c) the receiver-split numbers quoted in Section 3.3 prose (accepting
    42.7 -> 52.0 under the UG market frame; TG 53.0 -> 54.0 returning;
    expectation updated 2026-07-20 with the prose fix 42.6 -> 42.7, and
    since 2026-07-22 the same splits are displayed in tab:receiver_splits).
The OLS is computed with raw numpy linear algebra (no statsmodels).

Run: python3 verify_round2_similarity.py   (prints PASS/FAIL per block)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent          # replication_package/
PROJ = PKG.parent                 # project root
R2 = PROJ / "llm_similarity" / "round2"

CAT = {"M": "Moral", "S": "Self-interest", "C": "Mutual Benefit / Cooperation"}
CATCOLS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
ROW = {"BONUS": "Bonus story", "AID": "Aid story",
       "C-KW": "DG-KW Control", "M-KW": "DG-KW Market",
       "C-LT": "DG-LT Control", "M-LT": "DG-LT Market",
       "C-UG": "UG Control", "M-UG": "UG Market",
       "C-TG": "TG Control", "M-TG": "TG Market"}

n_fail = 0


def ok(name: str, cond: bool) -> None:
    global n_fail
    print(("  PASS  " if cond else "  FAIL  ") + name)
    n_fail += 0 if cond else 1


def parse_published_table() -> dict:
    """Parse the 10x6 numeric body of similarity_categories.tex."""
    tex = (PKG / "output" / "tables" / "similarity_categories.tex").read_text()
    out = {}
    for line in tex.splitlines():
        line = line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        if len(parts) == 7 and re.match(r"^-?\d+\.\d$", parts[1]):
            out[parts[0]] = [float(x) for x in parts[1:]]
    assert len(out) == 10, f"parsed {len(out)} rows, expected 10"
    return out


def main() -> None:
    ratings = pd.read_csv(R2 / "recording_round2.csv")
    splits = pd.read_csv(R2 / "recording_splits_round2.csv")
    ratings = ratings[ratings.model == "claude"]
    splits = splits[splits.model == "claude"]
    assert sorted(ratings["set"].unique()) == [1, 2, 3]
    assert len(ratings) == 3 * 10 * 24, f"unexpected rating count {len(ratings)}"

    cat_of = lambda v: CAT[v[0]]

    # --- Table 6, similarity block: mean of the 24 raw ratings per context x category
    sim = {}
    for (ctx, v), grp in ratings.groupby(["context", "vignette_id"]):
        sim.setdefault(ctx, {}).setdefault(cat_of(v), []).extend(grp.similarity_rating.tolist())
    sim_cell = {ctx: {c: np.mean(vals) for c, vals in d.items()} for ctx, d in sim.items()}

    # --- Table 6, retrieval block: category share of the retrieval points, pooled sets
    ret = splits[splits.task == "retrieval"].copy()
    assert len(ret) == 3 * 10 * 24
    ret["cat"] = ret.item_id.str[0].map(CAT)
    ret_cell = {}
    for ctx, d in ret.groupby("context"):
        tot = d.points.sum()
        ret_cell[ctx] = {c: 100.0 * d.loc[d.cat == c, "points"].sum() / tot for c in CATCOLS}

    pub = parse_published_table()
    max_dev = 0.0
    for ctx, label in ROW.items():
        mine = [sim_cell[ctx][c] for c in CATCOLS] + [ret_cell[ctx][c] for c in CATCOLS]
        dev = max(abs(round(m, 1) - p) for m, p in zip(mine, pub[label]))
        max_dev = max(max_dev, dev)
        ok(f"Table 6 row '{label}': recomputed {[round(m,1) for m in mine]} == published", dev < 1e-9)
    print(f"  [info] max |recomputed - published| over 60 cells: {max_dev}")

    # --- receiver splits quoted in Section 3.3 prose
    rec = splits[splits.task == "receiver"].copy()
    def rec_share(ctx: str, prefix: str) -> float:
        d = rec[rec.context == ctx]
        return 100.0 * d.loc[d.item_id.str.startswith(prefix), "points"].sum() / d.points.sum()
    ug_c, ug_m = rec_share("C-UG", "UA"), rec_share("M-UG", "UA")
    tg_c, tg_m = rec_share("C-TG", "TR"), rec_share("M-TG", "TR")
    ok(f"prose: UG accepting split 42.7 -> 52.0 (got {ug_c:.1f} -> {ug_m:.1f})",
       round(ug_c, 1) == 42.7 and round(ug_m, 1) == 52.0)
    ok(f"prose: TG returning split 53.0 -> 54.0 (got {tg_c:.1f} -> {tg_m:.1f})",
       round(tg_c, 1) == 53.0 and round(tg_m, 1) == 54.0)

    # --- Figure 6: 18 cells, dq (classified shares) vs ds (similarity), OLS by hand
    p1 = pd.read_excel(PKG / "data" / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p1 = p1[p1.category.isin(CATCOLS)]
    share = (p1.groupby(["game", "story"]).category
             .value_counts(normalize=True).rename("q").reset_index())
    q = {(r.game, int(r.story), r.category): r.q for r in share.itertuples()}

    xs, ys = [], []
    for game, ctx_c, ctx_m in [("dgkw", "C-KW", "M-KW"), ("ug", "C-UG", "M-UG"), ("tg", "C-TG", "M-TG")]:
        for c in CATCOLS:
            xs.append(sim_cell[ctx_m][c] - sim_cell[ctx_c][c])
            ys.append(q.get((game, 1, c), 0.0) - q.get((game, 0, c), 0.0))
    for game in ["dgkw", "ug", "tg"]:
        for c in CATCOLS:
            xs.append(sim_cell["AID"][c] - sim_cell["BONUS"][c])
            ys.append(q.get((game, 4, c), 0.0) - q.get((game, 2, c), 0.0))
    x, y = np.array(xs), np.array(ys)
    assert len(x) == 18

    X = np.column_stack([np.ones(18), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    rss, tss = resid @ resid, ((y - y.mean()) ** 2).sum()
    se_slope = np.sqrt(rss / (18 - 2) * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - rss / tss
    rho = stats.spearmanr(x, y).statistic
    print(f"  [info] Figure 6 recomputed: slope {beta[1]:.4f}, se {se_slope:.4f}, "
          f"R2 {r2:.3f}, Spearman {rho:.2f}")
    ok("Figure 6 slope 0.0122", round(beta[1], 4) == 0.0122)
    ok("Figure 6 se 0.0025", round(se_slope, 4) == 0.0025)
    ok("Figure 6 R2 0.59", round(r2, 2) == 0.59)
    ok("Figure 6 Spearman 0.57", round(rho, 2) == 0.57)
    ok("prose: TG market similarity shifts at most +1.5",
       max(abs(sim_cell["M-TG"][c] - sim_cell["C-TG"][c]) for c in CATCOLS) <= 1.5)

    print(f"\n{'ALL CHECKS PASS' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")


if __name__ == "__main__":
    main()
