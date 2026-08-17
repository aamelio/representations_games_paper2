"""Dry run of the equal-payoff norm-anchor amendment (held item 2 follow-up; HELD, no
paper changes): redo the TG calibration under the amended sender FOC and check that every
downstream object still goes through.

Amendment: the TG norm anchor moves from the equal-split action (1/2) to the send that
equalizes EXPECTED payoffs at the believed returned share s:
    t*(s) = 1 / (1 + (1+r)(1-2s)),   dt*/ds = 2(1+r) / (1 + (1+r)(1-2s))^2,
so the sender FOC becomes  t = t*(s) + [rho r - sigma(1+r)(1-s)] / mu, and the TG locus
    rho/mu = [t_TG - t*(s_hp) + (sigma/mu)(1+r)(1-s_hp)] / r.
DG and UG are untouched (there the equal-payoff action IS 1/2). Predicted TG belief
sensitivity gains a weight-free norm-channel term: dt/ds = dt*/ds + (1+r)(sigma/mu).

Everything else is 08_calibration.py verbatim (imported; only tg_locus monkey-patched).
Checks: (i) calibrated ratios old vs new, all categories; (ii) SI schedule validity
(a>=0, b>0, a+b<=1) and Moral non-identification; (iii) MBC joint solve convergence;
(iv) overidentification (chosen-action optimism, the 16-20pp claim); (v) predicted vs
observed TG slopes and the attenuation pattern; (vi) the coupling result
E[t]-E_indep[t] = (1+r)Cov(sigma/mu, s) (unchanged analytically: the t*(s) term is
additive in s, so it drops out of the independence comparison); (vii) t* censoring
(t*<=1 iff s<=1/2) at category means and individually; (viii) bootstrap SEs under the
amended locus (same B, seed).

Output: output/tg_anchor_dryrun_stats.txt
"""

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output"

spec = importlib.util.spec_from_file_location("calib08", HERE / "08_calibration.py")
c8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c8)

R = c8.R
CATS = c8.CATS
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def tstar(s):
    return 1.0 / (1.0 + (1 + R) * (1 - 2 * s))


def dtstar(s):
    return 2 * (1 + R) / (1.0 + (1 + R) * (1 - 2 * s)) ** 2


def tg_locus_amended(m, x):
    return (m["t_tg"] - tstar(m["s_hp"]) + x * (1 + R) * (1 - m["s_hp"])) / R


df = c8.load_control()
m_all = c8.moments(df)
slopes = c8.empirical_slopes(df)

base = c8.calibrate(m_all)
cov_b = c8.covariance_exercise(df, base)

c8.tg_locus = tg_locus_amended  # the single change
amend = c8.calibrate(m_all)
for cat in CATS:
    amend[cat]["sens_tg_pred"] = (1 + R) * amend[cat]["x"] + dtstar(m_all[cat]["s_hp"])
cov_a = c8.covariance_exercise(df, amend)

log("=== (i) calibrated ratios, baseline -> amended ===")
for cat in CATS:
    b, a = base[cat], amend[cat]
    s = m_all[cat]["s_hp"]
    log(f"{cat}: s_hp={s:.3f}, t*(s_hp)={tstar(s):.3f} | "
        f"sigma/mu {b['x']:.3f} -> {a['x']:.3f} | rho/mu {b['y']:.3f} -> {a['y']:.3f}")

log("\n=== (ii) UG schedule (a,b) and validity ===")
for cat in ["Moral", "Self-interest"]:
    b, a = base[cat], amend[cat]
    log(f"{cat}: (a,b) {b['a_foc']:.3f},{b['b_foc']:.3f} valid={b['ab_valid']} -> "
        f"{a['a_foc']:.3f},{a['b_foc']:.3f} valid={a['ab_valid']} "
        f"(a+b: {b['a_foc']+b['b_foc']:.3f} -> {a['a_foc']+a['b_foc']:.3f})")

log("\n=== (iii) MBC joint solve ===")
log(f"baseline x={base['Mutual Benefit / Cooperation']['x']:.4f}, "
    f"amended x={amend['Mutual Benefit / Cooperation']['x']:.4f}; "
    "rho>sigma on locus: "
    f"baseline {base['Mutual Benefit / Cooperation']['y'] > base['Mutual Benefit / Cooperation']['x']}, "
    f"amended {amend['Mutual Benefit / Cooperation']['y'] > amend['Mutual Benefit / Cooperation']['x']}")

log("\n=== (iv) overidentification: chosen-action optimism (pp) ===")
for cat in CATS:
    b, a = base[cat], amend[cat]
    gb = (b["p_chosen_actual"] - b["p_chosen_pred"]) * 100
    ga = (a["p_chosen_actual"] - a["p_chosen_pred"]) * 100
    log(f"{cat}: measured - implied = {gb:+.1f}pp -> {ga:+.1f}pp")

log("\n=== (v) TG belief sensitivity: predicted vs observed ===")
log(f"{'category':<30} {'pred old':>9} {'pred new':>9} {'observed':>9} "
    f"{'obs/old':>8} {'obs/new':>8}")
for cat in CATS:
    po = (1 + R) * base[cat]["x"]
    pn = amend[cat]["sens_tg_pred"]
    ob = slopes[("tg", cat)][0]
    log(f"{cat:<30} {po:9.3f} {pn:9.3f} {ob:9.3f} "
        f"{ob/po if po > 0 else np.inf:8.2f} {ob/pn:8.2f}")
ug_ratio = slopes[("ug", "Self-interest")][0] / -base["Self-interest"]["sens_ug_pred"]
log(f"(UG SI attenuation, unchanged by the amendment: {abs(ug_ratio):.2f})")

log("\n=== (vi) attention-belief coupling ===")
log(f"baseline: Cov={cov_b[0]:+.4f}, (1+r)Cov={cov_b[1]:+.4f} | "
    f"amended: Cov={cov_a[0]:+.4f}, (1+r)Cov={cov_a[1]:+.4f}")
log("(formula unchanged analytically: t*(s) is additive in s, drops out of the "
    "independent-marginals comparison; numeric shift only via the SI/MBC x's)")

log("\n=== (vii) t* censoring ===")
tg_beliefs = df[(df.game == "tg")].dropna(subset=["beliefs_hp"])
sh = (tg_beliefs.beliefs_hp >= 0.5).mean()
log(f"category-mean beliefs all < 1/2: "
    f"{all(m_all[c]['s_hp'] < 0.5 for c in CATS)}; individual TG beliefs >= 1/2: "
    f"{sh*100:.1f}% (their anchor censors at t*=1; footnote material, no hurdle)")

log("\n=== (viii) bootstrap under the amended locus ===")
ses, n_ok = c8.bootstrap(df)
log(f"{n_ok}/{c8.B_BOOT} successful draws (seed {c8.SEED})")
for cat in CATS:
    log(f"{cat}: se(sigma/mu)={ses.get((cat, 'x'), np.nan):.3f}, "
        f"se(rho/mu)={ses.get((cat, 'y'), np.nan):.3f}")

(OUT / "tg_anchor_dryrun_stats.txt").write_text("\n".join(LOG) + "\n")
print("\nwrote output/tg_anchor_dryrun_stats.txt")
