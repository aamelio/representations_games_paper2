"""Held item 2 support: can misclassified cooperators explain the TG Moral belief-slope
anomaly (observed 0.306 vs predicted 0.113)? HELD for the coauthor call, not in the paper.

Three parts.
A. DG detector: the memory measure (pre-decision, new scheme, 15_hp_person_level data)
   flags reasons-Moral participants whose retrieved situation is NOT moral. If cross-measure
   disagreement marks misclassification, "mixed" Morals should transfer less than "pure"
   Morals (and symmetrically for Self-interest).
B. Contamination rates: P(memory = MBC | reasons = Moral) etc., the direct DG estimate of
   the confusion the TG story requires (TG MBC base rate 37% vs 1-4% in DG, so DG rates
   are a lower bound for TG).
C. TG mixture arithmetic: the contamination share pi needed to lift a true Moral slope
   (predicted 0.113, or attenuation-adjusted using the SI ratio) to the observed 0.306,
   for contaminant slopes equal to the observed MBC and SI cell slopes. Compares pi_needed
   with the DG-estimated rates. Also recomputes the TG control slopes from the microdata
   (same spec as 05: OLS share_sent ~ beliefs_hp, HC1, classified, control) as a check
   against tab:belief_sensitivity.

Output: output/moral_slope_check_stats.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "output"

CATS = ["Moral", "Mutual Benefit / Cooperation", "Self-interest"]
PRED = {"Moral": 0.113, "Self-interest": 1.192,
        "Mutual Benefit / Cooperation": 0.000}  # calibration_stats.txt, TG
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def welch(a, b):
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p = 2 * stats.norm.sf(abs(d / se))
    return d, se, p


hp = pd.read_excel(DATA / "hpmin_new_scheme_categorized.xlsx")
p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
hp["story"] = hp.Market
m = hp.merge(p1, on=["PROLIFIC_PID", "game", "story"], how="left",
             suffixes=("_hp", "_re"))
assert len(m) == len(hp) and m.category_re.notna().all()

# ---- A. DG detector: pure vs mixed within reasons category (control) ------------------
log("=== A. DG detector: transfers by memory agreement within reasons category ===")
for cond, cname in [(0, "Control"), (1, "Market")]:
    d = m[m.story == cond]
    log(f"\n-- {cname} --")
    for re_cat, mixed_cats in [("Moral", ["Self-interest", "Mutual Benefit / Cooperation"]),
                               ("Self-interest", ["Moral", "Mutual Benefit / Cooperation"])]:
        g = d[d.category_re == re_cat]
        pure = g[g.category_hp == re_cat].share_sent
        mixed = g[g.category_hp.isin(mixed_cats)].share_sent
        noclr = g[g.category_hp == "No clear justification"].share_sent
        log(f"reasons = {re_cat}: pure {pure.mean():.3f} (N={len(pure)}) | "
            f"mixed {mixed.mean():.3f} (N={len(mixed)}) | "
            f"memory-unclassified {noclr.mean():.3f} (N={len(noclr)})")
        if len(pure) > 5 and len(mixed) > 5:
            dd, se, p = welch(pure, mixed)
            log(f"  pure - mixed = {dd:+.3f} (se {se:.3f}, p = {p:.3g})")
        if re_cat == "Moral":
            log(f"  mass at exactly 1/2: pure {(pure == 0.5).mean():.3f}, "
                f"mixed {(mixed == 0.5).mean():.3f}")

# ---- B. contamination rates from the DG cross-tab -------------------------------------
log("\n=== B. DG contamination rates (memory category within reasons = Moral) ===")
for cond, cname in [(0, "Control"), (1, "Market")]:
    g = m[(m.story == cond) & (m.category_re == "Moral")]
    shares = g.category_hp.value_counts(normalize=True) * 100
    log(f"{cname} (N={len(g)}): " + "; ".join(
        f"{c}: {shares.get(c, 0):.1f}%" for c in
        ["Moral", "Mutual Benefit / Cooperation", "Self-interest",
         "No clear justification"]))
log("DG MBC base rate (reasons, control): "
    f"{(m[(m.story == 0)].category_re == 'Mutual Benefit / Cooperation').mean() * 100:.1f}%"
    " -- TG base rate is 37%, so TG confusion should be higher than these DG rates.")

# ---- C. TG mixture arithmetic ---------------------------------------------------------
log("\n=== C. TG mixture arithmetic ===")
tg = p1[(p1.game == "tg") & (p1.story == 0) & p1.category.isin(CATS)].copy()
s = tg.beliefs_hp
if s.max() > 1.0:  # raw dollars back of tripled $6 -> share
    tg["beliefs_hp"] = s / 6
obs = {}
for cat in CATS:
    d = tg[(tg.category == cat) & tg.beliefs_hp.notna() & tg.share_sent.notna()]
    r = smf.ols("share_sent ~ beliefs_hp", data=d).fit(cov_type="HC1")
    obs[cat] = (r.params["beliefs_hp"], r.bse["beliefs_hp"], int(r.nobs))
    log(f"TG control slope, {cat}: {obs[cat][0]:.3f} (se {obs[cat][1]:.3f}, "
        f"N={obs[cat][2]}) [predicted {PRED[cat]:.3f}]")
atten = obs["Self-interest"][0] / PRED["Self-interest"]
log(f"attenuation factor (SI observed/predicted): {atten:.3f}")

s_moral_obs = obs["Moral"][0]
log("\npi needed so that observed Moral slope = (1-pi)*true + pi*contaminant:")
for true_lab, s_true in [("predicted 0.113", PRED["Moral"]),
                         (f"attenuated {atten:.2f}x0.113", atten * PRED["Moral"])]:
    for cont_lab, s_cont in [("observed MBC cell", obs["Mutual Benefit / Cooperation"][0]),
                             ("observed SI cell", obs["Self-interest"][0])]:
        pi = (s_moral_obs - s_true) / (s_cont - s_true) if s_cont != s_true else np.inf
        verdict = ("INFEASIBLE (>1)" if pi > 1 else
                   "implausible vs DG rates" if pi > 0.35 else "plausible")
        log(f"  true = {true_lab}, contaminant = {cont_lab}: "
            f"pi = {pi:.2f}  [{verdict}]")

log("\nNote: the MBC cell itself exceeds its prediction (obs "
    f"{obs['Mutual Benefit / Cooperation'][0]:.3f} vs 0.000), so 'excess' belief "
    "sensitivity is shared by Moral AND MBC senders. A belief-dependent moral/cooperative "
    "target (send so that payoffs equalize at the EXPECTED return, cf. the receiver-side "
    "payoff-equalization result in 13_receiver_models.py) predicts positive slopes in "
    "both cells with no misclassification needed.")

(OUT / "moral_slope_check_stats.txt").write_text("\n".join(LOG) + "\n")
print("\nwrote output/moral_slope_check_stats.txt")
