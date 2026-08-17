#!/usr/bin/env python3
"""M1 leave-one-out check — held OUT of the paper text for now (SN decision 2026-07-19).

M1 (workplace spot-bonus sharing) is the vignette closest in content to the Bonus story.
This script recomputes the context x category table and the H5 regression excluding M1,
separately for each rater model in the recording CSV (headline Opus rater and the Fable-5
robustness rater). The corresponding paper text (a clause in S3.3 and a sentence in
app:similarity_vignettes) is commented out in main.tex, not deleted; re-run this script
and uncomment to restore.

Output: out/loo_m1.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

CAT_OF = {"M": "Moral", "S": "Self-interest", "C": "Mutual Benefit / Cooperation"}
GAMES = [("dgkw", "KW"), ("ug", "UG"), ("tg", "TG")]

ratings = pd.read_csv(HERE / "recording_round2.csv", comment="#")
ratings["category"] = ratings["vignette_id"].str[0].map(CAT_OF)

p1 = pd.read_excel(HERE.parent.parent / "replication_package/data/player1_all_categorized.xlsx")
p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
p1 = p1[p1["category"].isin(CAT_OF.values())]
COND = {"control": 0, "market": 1, "bonus": 2, "aid": 4}
shares = (p1.groupby(["game", "story", "category"]).size()
          / p1.groupby(["game", "story"]).size()).rename("share").reset_index()


def share(game, cond, cat):
    r = shares.query("game == @game and story == @cond and category == @cat")["share"]
    return float(r.iloc[0]) if len(r) else np.nan


def h5_fit(byc):
    rows = []
    for comparison, c_hi, c_lo in [("Market-Control", COND["market"], COND["control"]),
                                   ("Aid-Bonus", COND["aid"], COND["bonus"])]:
        for game, ckey in GAMES:
            for cat in CAT_OF.values():
                dq = share(game, c_hi, cat) - share(game, c_lo, cat)
                if comparison == "Market-Control":
                    ds = byc.loc[f"M-{ckey}", cat] - byc.loc[f"C-{ckey}", cat]
                else:
                    ds = byc.loc["AID", cat] - byc.loc["BONUS", cat]
                rows.append(dict(dq=dq, ds=ds))
    h5 = pd.DataFrame(rows)
    return sm.OLS(h5["dq"], sm.add_constant(h5["ds"]), missing="drop").fit()


lines = []
for model in sorted(ratings["model"].unique()):
    rm = ratings[ratings.model == model]
    out = {}
    for tag, rr in [("full", rm), ("no_M1", rm[rm.vignette_id != "M1"])]:
        byc = rr.groupby(["context", "category"])["similarity_rating"].mean().unstack()
        fit = h5_fit(byc)
        out[tag] = (byc, fit)
        lines.append(f"{model} [{tag}]: Bonus-Moral {byc.loc['BONUS', 'Moral']:.1f}, "
                     f"Aid-Moral {byc.loc['AID', 'Moral']:.1f}; H5 slope "
                     f"{fit.params.iloc[1]:.4f} (se {fit.bse.iloc[1]:.4f}), "
                     f"R2 {fit.rsquared:.3f}")
    dfull = (out["no_M1"][0] - out["full"][0]).abs()
    lines.append(f"{model}: max |cell change| outside Bonus-Moral = "
                 f"{dfull.drop(index='BONUS').max().max():.1f}\n")

Path(OUT / "loo_m1.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
