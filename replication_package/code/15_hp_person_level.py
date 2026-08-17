"""P6 (NG p.22) person-level consistency check: the hp-allocation representation measure
vs the reasons-based measure, within participant.

Inputs: data/hpmin_new_scheme_categorized.xlsx (from 14_hp_classification.py) merged with
data/player1_all_categorized.xlsx on (PROLIFIC_PID, game, story) -- exact 1:1, verified.
The hp row is the remembered situation whose remembered allocation ("hp", dollars kept of
$12) is closest to the participant's actual allocation (AA's hpmin selection); the memory
was elicited BEFORE the transfer decision, the reasons question after it.

Caveat carried into all outputs: the hpmin selection makes memory-NUMBER vs behavior
relations partly mechanical; the informative person-level content is the association of
the memory's TEXT classification (category, social proximity) with the reasons measure
and with behavior.

Outputs: output/hp_person_level_stats.txt + output/tables/hp_person_level.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "output"

HIGH_SP = ["Anonymous peer", "Teammate / coworker", "Friend"]  # sp_num >= 2, as Table 2
CATS = ["Moral", "Mutual Benefit / Cooperation", "Self-interest"]
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


hp = pd.read_excel(DATA / "hpmin_new_scheme_categorized.xlsx")
p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
hp["story"] = hp.Market
m = hp.merge(p1, on=["PROLIFIC_PID", "game", "story"], how="left",
             suffixes=("_hp", "_re"))
assert m.category_re.notna().all() and len(m) == len(hp)
m["high_sp_hp"] = m.social_proximity_hp.isin(HIGH_SP)
m["high_sp_re"] = m.social_proximity_re.isin(HIGH_SP)

log(f"N = {len(m)} participant-frame observations (DG-KW/DG-LT, Control + Market);"
    f" memory classified: {m.category_num_hp.notna().sum()}")

# -- 1. category consistency: memory (pre-decision) vs reasons (post-decision) ----------
both = m[m.category_hp.isin(CATS) & m.category_re.isin(CATS)].copy()
log(f"\n[1] Category cross-tab, substantive categories both measures (N={len(both)}):")
ct = pd.crosstab(both.category_re, both.category_hp)
log(ct.to_string())
a = both.category_re.to_numpy()
b = both.category_hp.to_numpy()
agree = (a == b).mean()
pe = sum((a == k).mean() * (b == k).mean() for k in CATS)
kappa = (agree - pe) / (1 - pe)
chi2, p, _, _ = stats.chi2_contingency(ct)
V = np.sqrt(chi2 / (len(both) * (min(ct.shape) - 1)))
log(f"agreement {agree:.3f}; Cohen's kappa {kappa:.3f}; "
    f"chi2({(ct.shape[0]-1)*(ct.shape[1]-1)}) = {chi2:.1f}, p = {p:.2g}; "
    f"Cramer's V = {V:.3f}")
for cond, lab in [(0, "Control"), (1, "Market")]:
    d = both[both.story == cond]
    log(f"  {lab}: N={len(d)}, agreement {(d.category_re == d.category_hp).mean():.3f}")
log("Moral-memory share by reasons category (column %):")
log((ct.div(ct.sum(1), axis=0) * 100).round(1).to_string())

# -- 2. SP consistency (NG's 'share lunch with my friend') ------------------------------
log("\n[2] Social proximity, memory vs reasons (all classified rows):")
sp_ct = pd.crosstab(m.high_sp_re, m.high_sp_hp)
log(sp_ct.to_string())
phi = stats.pearsonr(m.high_sp_re.astype(int), m.high_sp_hp.astype(int))
log(f"phi = {phi.statistic:.3f}, p = {phi.pvalue:.2g}")
mo = m[m.category_re == "Moral"].high_sp_hp
si = m[m.category_re == "Self-interest"].high_sp_hp
z2 = stats.norm.sf(abs((mo.mean() - si.mean()) /
                       np.sqrt(mo.mean() * (1 - mo.mean()) / len(mo) +
                               si.mean() * (1 - si.mean()) / len(si)))) * 2
log(f"high-SP memory share: Moral reasons {mo.mean():.3f} (N={len(mo)}) vs "
    f"Self-interest reasons {si.mean():.3f} (N={len(si)}), two-prop z p = {z2:.2g}")

# -- 3. behavior: allocation by memory category (gradient direction) --------------------
log("\n[3] Share sent by memory category (control; hpmin caveat applies to levels):")
ctl = m[(m.story == 0) & m.category_hp.isin(CATS)]
log(ctl.groupby("category_hp").share_sent.agg(["mean", "size"]).round(3).to_string())
log("Reasons-category gradient, same sample:")
log(ctl[ctl.category_re.isin(CATS)].groupby("category_re")
    .share_sent.agg(["mean", "size"]).round(3).to_string())

# -- 4. market effect on the memory classification (new scheme, pre-decision) -----------
log("\n[4] Memory categories by condition (new scheme, column %):")
mc = pd.crosstab(m.category_hp, m.story, normalize="columns") * 100
mc.columns = ["Control", "Market"]
log(mc.round(1).to_string())

(OUT / "hp_person_level_stats.txt").write_text("\n".join(LOG) + "\n")

# -- LaTeX table: joint distribution + consistency stats --------------------------------
short = {"Moral": "Moral", "Mutual Benefit / Cooperation": "MBC",
         "Self-interest": "Self-interest"}
rows = []
colshare = ct.div(ct.sum(1), axis=0) * 100
for re_cat in CATS:
    cells = " & ".join(f"{colshare.loc[re_cat, c]:.1f}\\%" for c in CATS)
    rows.append(f"{short[re_cat]} & {cells} & {ct.loc[re_cat].sum()} \\\\")
tex = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Person-level consistency of the two representation measures: distribution "
    r"of the hypothetical-allocation (memory) category within each stated-reason "
    r"category, dictator games, Control and Market conditions. The memory measure is "
    r"elicited before the transfer decision and classified with the same category scheme "
    r"as the reasons; rows sum to 100\%.}",
    r"\label{tab:hp_person_level}",
    r"\begin{tabular}{lcccc}", r"\toprule",
    r" & \multicolumn{3}{c}{Memory (pre-decision) category} & \\",
    r"\cmidrule(lr){2-4}",
    r"Reasons category & Moral & MBC & Self-interest & $N$ \\ \midrule",
    *rows,
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    rf"\footnotesize Notes: $N={len(both)}$ participant-frame observations with "
    rf"substantive categories on both measures. Agreement {agree*100:.1f}\%, Cohen's "
    rf"$\kappa={kappa:.2f}$, Cram\'er's $V={V:.2f}$ "
    rf"($\chi^2$ $p<{max(p, 0.001):.3f}$). The memory retained for each participant is "
    r"the one attached to the hypothetical allocation closest to the actual allocation, "
    r"so memory--behavior comparisons are partly mechanical; the category and "
    r"social-proximity content of the memory text is not.",
    r"\end{flushleft}", r"\end{table}", "",
]
(OUT / "tables" / "hp_person_level.tex").write_text("\n".join(tex))
print("\nwrote output/hp_person_level_stats.txt, output/tables/hp_person_level.tex")
