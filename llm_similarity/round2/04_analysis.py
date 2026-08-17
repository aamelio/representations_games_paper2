#!/usr/bin/env python3
"""Similarity round 2 - analysis.

Inputs: recording_round2.csv, recording_splits_round2.csv (from 03_api_runner.py or manual entry
into the *_template.csv schemas), and ../../replication_package/data/player1_all_categorized.xlsx
(+ player2 for receiver shifts).

Outputs (to out/):
  contexts_by_category.csv     context x category mean similarity (over models, sets, vignettes)
  retrieval_by_category.csv    context x category mean retrieval points
  receiver_splits.csv          strategic context x counterpart type mean points
  h5_scatter.png + h5_stats.txt  Dq_c (measured category-share shifts) vs DS (similarity shifts),
                               the 6 game x comparison cells x 3 categories; NG's regression
  range_check.txt              per-cell ranges across models and sets (robustness, May style)

Run after the rating round completes. Population shares from the classified sample, control/market
and bonus/aid conditions, substantive categories only (matches the paper's figures convention).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

CAT_OF = {"M": "Moral", "S": "Self-interest", "C": "Mutual Benefit / Cooperation"}
GAME_OF_CONTEXT = {"C-KW": "DG-KW", "M-KW": "DG-KW", "C-LT": "DG-LT", "M-LT": "DG-LT",
                   "C-UG": "UG", "M-UG": "UG", "C-TG": "TG", "M-TG": "TG"}

ratings = pd.read_csv(HERE / "recording_round2.csv", comment="#")
splits = pd.read_csv(HERE / "recording_splits_round2.csv", comment="#")
ratings["category"] = ratings["vignette_id"].str[0].map(CAT_OF)

# headline rater = Opus 4.8 ("claude"); any other models in the CSVs (e.g. the Fable-5
# robustness run) enter the range check only, never the paper exhibits
HEADLINE_MODEL = "claude"
PKG = HERE.parent.parent / "replication_package" / "output"
ratings_h = ratings[ratings["model"] == HEADLINE_MODEL]
splits_h = splits[splits["model"] == HEADLINE_MODEL]

# -- context x category means ------------------------------------------------
byc = (ratings_h.groupby(["context", "category"])["similarity_rating"]
       .mean().unstack().round(1))
byc.to_csv(OUT / "contexts_by_category.csv")
print("context x category similarity:\n", byc, "\n")

retr = splits_h.query("task == 'retrieval'").copy()
retr["category"] = retr["item_id"].str[0].map(CAT_OF)
byr = retr.groupby(["context", "category"])["points"].sum()
byr = (byr / retr.groupby("context")["points"].sum()).unstack().mul(100).round(1)
byr.to_csv(OUT / "retrieval_by_category.csv")

recv = splits_h.query("task == 'receiver'").copy()
recv["type"] = recv["item_id"].str[:2]
byrec = recv.groupby(["context", "type"])["points"].mean().unstack().round(1)
byrec.to_csv(OUT / "receiver_splits.csv")
print("receiver splits:\n", byrec, "\n")

# -- paper table: similarity + retrieval by context x category ---------------
ROW_LABEL = [("BONUS", "Bonus story"), ("AID", "Aid story"),
             ("C-KW", "DG-KW Control"), ("M-KW", "DG-KW Market"),
             ("C-LT", "DG-LT Control"), ("M-LT", "DG-LT Market"),
             ("C-UG", "UG Control"), ("M-UG", "UG Market"),
             ("C-TG", "TG Control"), ("M-TG", "TG Market")]
CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
tex = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
       r"\renewcommand{\arraystretch}{1.15}",
       r"\caption{\textbf{Similarity of Contexts to Representation Categories, LLM Raters}}",
       r"\label{tab:similarity_categories}",
       r"\begin{tabular}{l ccc ccc}", r"\toprule",
       r"& \multicolumn{3}{c}{Similarity (0--100)} & \multicolumn{3}{c}{Retrieval split (\%)} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
       r"Context & Moral & Self-int. & Mut.\ Ben. & Moral & Self-int. & Mut.\ Ben. \\",
       r"\midrule"]
for key, lab in ROW_LABEL:
    sim = " & ".join(f"{byc.loc[key, c]:.1f}" for c in CATS)
    ret = " & ".join(f"{byr.loc[key, c]:.1f}" for c in CATS)
    tex.append(f"{lab} & {sim} & {ret} \\\\")
    if key == "AID":
        tex.append(r"\midrule")
tex += [r"\bottomrule", r"\end{tabular}", r"\begin{flushleft}",
        r"\footnotesize Notes: Similarity: 0--100 rating of each context text against eight"
        r" vignette exemplars per category, judging the situation as a whole (stakes,"
        r" relationship, setting, and decision structure), averaged over vignettes and three"
        r" permuted-label conversations (Claude Opus 4.8 raters with reasoning enabled;"
        r" single-model working measurement). Retrieval split: 100 points distributed across"
        r" the 24 vignettes"
        r" according to which would come to mind as a similar experience, aggregated by"
        r" category. Context texts are verbatim from the instructions and stories"
        r" (Appendix~\ref{app:instructions}). Design, prompts, and robustness:"
        r" Appendix~\ref{app:similarity_vignettes}.",
        r"\end{flushleft}", r"\end{table}", ""]
(PKG / "tables" / "similarity_categories.tex").write_text("\n".join(tex))

# -- robustness: ranges across models and sets (May convention) --------------
with open(OUT / "range_check.txt", "w") as f:
    for (lvl, name) in [("model", "across models"), ("set", "across label sets")]:
        cell = (ratings.groupby(["context", "category", lvl])["similarity_rating"].mean()
                .groupby(["context", "category"]).agg(np.ptp))
        f.write(f"max range {name}: {cell.max():.1f}\n{cell.round(1).to_string()}\n\n")

# -- H5: measured representation shifts vs similarity shifts -----------------
p1 = pd.read_excel(HERE.parent.parent / "replication_package/data/player1_all_categorized.xlsx")
p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
p1 = p1[p1["category"].isin(CAT_OF.values())]  # substantive categories only
COND = {"control": 0, "market": 1, "bonus": 2, "aid": 4}  # STORY_LABELS in pipeline script 01

shares = (p1.groupby(["game", "story", "category"]).size()
          / p1.groupby(["game", "story"]).size()).rename("share").reset_index()


def share(game, cond, cat):
    r = shares.query("game == @game and story == @cond and category == @cat")["share"]
    return float(r.iloc[0]) if len(r) else np.nan


rows = []
for comparison, c_hi, c_lo in [("Market-Control", COND["market"], COND["control"]),
                               ("Aid-Bonus", COND["aid"], COND["bonus"])]:
    for game, ckey in [("dgkw", "KW"), ("ug", "UG"), ("tg", "TG")]:
        for cat in CAT_OF.values():
            dq = share(game, c_hi, cat) - share(game, c_lo, cat)
            if comparison == "Market-Control":
                ds = byc.loc[f"M-{ckey}", cat] - byc.loc[f"C-{ckey}", cat]
            else:  # stories: same two contexts for every game; game enters via dq only
                ds = byc.loc["AID", cat] - byc.loc["BONUS", cat]
            rows.append(dict(comparison=comparison, game=game, category=cat, dq=dq, ds=ds))
h5 = pd.DataFrame(rows)

import statsmodels.api as sm  # noqa: E402

fit = sm.OLS(h5["dq"], sm.add_constant(h5["ds"]), missing="drop").fit()
with open(OUT / "h5_stats.txt", "w") as f:
    f.write(h5.round(3).to_string(index=False) + "\n\n")
    f.write(f"OLS dq on ds: slope {fit.params.iloc[1]:.4f} (se {fit.bse.iloc[1]:.4f}), "
            f"R2 {fit.rsquared:.3f}, N {int(fit.nobs)}\n")
    f.write(f"Spearman rho: {h5[['dq', 'ds']].corr(method='spearman').iloc[0, 1]:.3f}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 2026-07-22 (SN): color = category, marker = comparison, text label = game.
# The Aid-Bonus points stack at three x values by construction (story texts are
# game-invariant, so each category has a single story similarity shift).
CAT_COLOR = {"Moral": "#0072B2", "Self-interest": "#D55E00",
             "Mutual Benefit / Cooperation": "#009E73"}
CAT_LEGEND = {"Moral": "Moral", "Self-interest": "Self-interest",
              "Mutual Benefit / Cooperation": "Mutual Benefit/Coop."}
GAME_TAG = {"dgkw": "KW", "ug": "UG", "tg": "TG"}
LABEL_XY = {  # annotation offsets (points) tuned against overlaps; default (5, 2)
    ("Aid-Bonus", "dgkw", "Moral"): (5, -9),
    ("Aid-Bonus", "tg", "Self-interest"): (-21, -4),
    ("Aid-Bonus", "dgkw", "Self-interest"): (5, 4),
    ("Aid-Bonus", "ug", "Self-interest"): (5, -10),
    ("Aid-Bonus", "tg", "Mutual Benefit / Cooperation"): (5, 7),
    ("Aid-Bonus", "dgkw", "Mutual Benefit / Cooperation"): (5, -2),
    ("Aid-Bonus", "ug", "Mutual Benefit / Cooperation"): (5, -11),
}

fig, ax = plt.subplots(figsize=(6.4, 5.2))
for comp, mk in [("Market-Control", "o"), ("Aid-Bonus", "s")]:
    d = h5.query("comparison == @comp")
    ax.scatter(d["ds"], d["dq"], marker=mk, s=46,
               c=[CAT_COLOR[c] for c in d["category"]],
               edgecolor="black", linewidth=0.5, zorder=3)
    for r in d.itertuples():
        dx, dy = LABEL_XY.get((comp, r.game, r.category), (5, 2))
        ax.annotate(GAME_TAG[r.game], (r.ds, r.dq), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color="#444444")
xs = np.linspace(h5["ds"].min(), h5["ds"].max(), 10)
ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, lw=1, color="gray")
ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
ax.set_xlabel("Δ similarity of context to category (LLM raters)")
ax.set_ylabel("Δ category share (classified reasons)")
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                  markerfacecolor=CAT_COLOR[c], markeredgecolor="black",
                  label=CAT_LEGEND[c]) for c in CAT_OF.values()]
handles += [Line2D([0], [0], marker=mk, linestyle="none", markersize=7,
                   markerfacecolor="#cccccc", markeredgecolor="black", label=comp)
            for comp, mk in [("Market-Control", "o"), ("Aid-Bonus", "s")]]
ax.legend(handles=handles, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "h5_scatter.png", dpi=200)
fig.savefig(PKG / "figures" / "similarity_h5.pdf")

# M1 leave-one-out: moved to 05_loo_m1.py and HELD OUT of the paper text (SN decision
# 2026-07-19); the corresponding main.tex passages are commented out, not deleted.

# -- within-provider robustness: Fable-5 rater vs the headline rater ---------
fab = ratings[ratings["model"] == "claude_fable"]
if len(fab):
    fabc = fab.groupby(["context", "category"])["similarity_rating"].mean().unstack()
    both = pd.concat([byc.stack(), fabc.stack()], axis=1, join="inner",
                     keys=["opus", "fable"])
    d = (both["opus"] - both["fable"]).abs()
    ns = fab["set"].nunique()
    note = ("all 3 permuted conversations completed; the third required substitute "
            "permutations (sets 4-6 in 03_api_runner.py) after stochastic safety-filter "
            "refusals of the opening message"
            if ns == 3 else
            f"{ns} of 3 permuted conversations completed; the remainder declined at the "
            "outset by the model's safety filter and not retried")
    with open(OUT / "fable_check.txt", "w") as f:
        f.write(f"Fable-5 rater ({note}) vs headline Opus 4.8 rater, "
                "context x category cell means:\n")
        f.write(f"Pearson r {np.corrcoef(both['opus'], both['fable'])[0, 1]:.3f}; "
                f"mean |diff| {d.mean():.1f}; max |diff| {d.max():.1f} at {d.idxmax()}\n")
        for g in ["KW", "LT", "UG", "TG"]:
            dm = fabc.loc[f"M-{g}", "Moral"] - fabc.loc[f"C-{g}", "Moral"]
            f.write(f"Fable Market-Control Moral shift, {g}: {dm:+.1f}\n")

# -- test-retest: v2 (craft-markets freeze) vs archived v1 run ---------------
v1csv = HERE / "v1_craftfairs" / "out" / "contexts_by_category.csv"
if v1csv.exists():
    v1 = pd.read_csv(v1csv, index_col=0)
    a, b = byc.stack(), v1.stack()
    both = pd.concat([a, b], axis=1, join="inner", keys=["v2", "v1"])
    diff = (both["v2"] - both["v1"]).abs()
    r = np.corrcoef(both["v2"], both["v1"])[0, 1]
    with open(OUT / "test_retest.txt", "w") as f:
        f.write("v2 (craft-markets freeze) vs v1 (craft-fairs pilot run), "
                "context x category cell means (30 cells):\n")
        f.write(f"Pearson r {r:.3f}; mean |diff| {diff.mean():.1f}; "
                f"max |diff| {diff.max():.1f} at {diff.idxmax()}\n")

print("wrote out/: contexts_by_category, retrieval_by_category, receiver_splits, h5_*, "
      "range_check, fable_check, test_retest; paper files: similarity_categories.tex, "
      "similarity_h5.pdf")
