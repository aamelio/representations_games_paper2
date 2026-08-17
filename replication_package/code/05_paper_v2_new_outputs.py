"""
05_paper_v2_new_outputs.py

New outputs for paper_v2.tex (NG-skeleton restructure, July 9), per exhibit_specs_v2.md:

E1. Action vs hypothetical belief by representation category (control, Player 1, UG & TG):
    figure + belief-sensitivity regressions (belief x category interaction).
E2. Control vs Market outcome distributions, three panels (DG-KW, UG, TG), common scales.
E3. Heterogeneity tied to representations: between/within-category variance decomposition
    by game x arm, and predicted-vs-actual dispersion change for the two preregistered
    comparisons (category-conditional distributions pooled across the two arms).
E4. KW-LT framing gap on the treatment scale (bootstrap CI) vs Aid-Bonus and Market effects.
E5. Similarity-gradient consistency: story effects by game vs LLM structural similarity
    (means from LLM_Similarity/memory_games_llm_recording.xlsx) and retrieval splits.

Existing tables and figures are NOT touched; all outputs are new files.
Every section prints validation lines against the v1.1 audited numbers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # replication_package/
P1_FILE = ROOT / "data" / "player1_all_categorized.xlsx"
FIG_DIR = ROOT / "output" / "figures"
TABLE_DIR = ROOT / "output" / "tables"
STATS_FILE = TABLE_DIR / "paper_v2_new_stats.txt"

# E1 drawn 10.5in wide and included at ~0.9\textwidth (5.85in), E2 drawn 12.5in and
# included at \textwidth (6.5in): draw-to-display ratios ~1.8-1.9, so scale fonts to
# land near 8.5-9pt on the page. See FONT_SCALE in scripts 01-04 (added 2026-07-16,
# same treatment as the 07-15 restyle of the other scripts).
FONT_SCALE = 1.6

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 11 * FONT_SCALE,
        "legend.fontsize": 9 * FONT_SCALE,
        "xtick.labelsize": 9 * FONT_SCALE,
        "ytick.labelsize": 9 * FONT_SCALE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STORY = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}
GAMES = ["dgkw", "dglt", "ug", "tg"]
GAME_LABELS = {"dgkw": "DG-KW", "dglt": "DG-LT", "ug": "UG", "tg": "TG"}
CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
CAT_SHORT = {"Moral": "Moral", "Self-interest": "Self-interest",
             "Mutual Benefit / Cooperation": "Mutual Benefit/Coop."}
# Identity colors separated primarily by lightness (dark / mid / light) so the
# distinction survives all CVD types; markers and linestyles add non-color encoding.
CAT_COLOR = {"Moral": "#4C566A", "Self-interest": "#C65D3A",
             "Mutual Benefit / Cooperation": "#8FBCBB"}
CAT_MARKER = {"Moral": "o", "Self-interest": "s", "Mutual Benefit / Cooperation": "^"}
CAT_LS = {"Moral": "-", "Self-interest": "--", "Mutual Benefit / Cooperation": "-."}
CONTROL_COLOR, MARKET_COLOR = "#4C566A", "#C65D3A"
# One color per game (matches GAME_COLORS in script 02) for the predicted-vs-actual
# dispersion scatter; one marker per comparison.
GAME_COLOR = {"dgkw": "#4C566A", "dglt": "#7B879D", "ug": "#D08770", "tg": "#8FBCBB"}
COMPARISON_MARKER = {"Market vs Control": "o", "Aid vs Bonus": "s"}

rng = np.random.default_rng(20260709)
stats_lines = []


def log(line=""):
    print(line)
    stats_lines.append(str(line))


def check(label, got, want, tol):
    flag = "OK " if abs(got - want) <= tol else "***MISMATCH***"
    log(f"  [validate] {label}: got {got:.3f}, v1 says {want:.3f}  {flag}")


# ---------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------

p1 = pd.read_excel(P1_FILE)
p1 = p1[p1.story.isin(STORY) & p1.game.isin(GAMES)].copy()
p1["story"] = p1["story"].astype(int)
p1["classified"] = p1.category.isin(CATS)

# Normalize hypothetical beliefs to shares/probabilities in [0,1].
log("=== Data preparation ===")
for g in ["ug", "tg"]:
    s = p1.loc[p1.game == g, "beliefs_hp"]
    mx = s.max()
    if mx <= 1.0:
        scale = 1.0
    elif mx <= 6.5:
        scale = 1 / 6  # TG: dollars back out of the tripled $6
    elif mx <= 100.0:
        scale = 1 / 100  # percent
    else:
        raise ValueError(f"unexpected beliefs_hp range in {g}: max={mx}")
    p1.loc[p1.game == g, "beliefs_hp"] = s * scale
    log(f"beliefs_hp {g}: raw max {mx:.2f} -> scale {scale:.4f}, "
        f"normalized mean {p1.loc[p1.game == g, 'beliefs_hp'].mean():.3f}, "
        f"N nonmissing {p1.loc[p1.game == g, 'beliefs_hp'].notna().sum()}")

ctrl = p1[p1.story == 0]
check("UG control mean hypothetical belief", ctrl.loc[ctrl.game == "ug", "beliefs_hp"].mean(), 0.483, 0.01)
for cat, want in [("Self-interest", 0.220), ("Moral", 0.315), ("Mutual Benefit / Cooperation", 0.357)]:
    got = ctrl.loc[(ctrl.game == "tg") & (ctrl.category == cat), "beliefs_hp"].mean()
    check(f"TG control belief, {cat}", got, want, 0.01)
check("DG-KW control mean share", ctrl.loc[ctrl.game == "dgkw", "share_sent"].mean(), 0.350, 0.005)
check("DG-LT control mean share", ctrl.loc[ctrl.game == "dglt", "share_sent"].mean(), 0.414, 0.005)
log()


def cell(game, story, col="share_sent", classified_only=False):
    d = p1[(p1.game == game) & (p1.story == story)]
    if classified_only:
        d = d[d.classified]
    return d[col].dropna()


def welch_diff(a, b):
    """b minus a, with Welch SE."""
    d = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, se


# ---------------------------------------------------------------------
# E1. Action vs hypothetical belief by category (control, P1, UG & TG)
# ---------------------------------------------------------------------

log("=== E1: action vs hypothetical belief by category ===")

BELIEF_LABEL = {"ug": "Believed P(one-third offer accepted)",
                "tg": "Believed return (share of tripled amount)"}
ACTION_LABEL = {"ug": "Offer (share of budget)", "tg": "Sent (share of endowment)"}

e1_rows = []  # (game, cat, sample, slope, se, n)
eq_tests = {}

for g in ["ug", "tg"]:
    for sample, dat in [("Control", p1[(p1.game == g) & (p1.story == 0)]),
                        ("Pooled", p1[p1.game == g])]:
        d = dat[dat.classified & dat.beliefs_hp.notna() & dat.share_sent.notna()].copy()
        d["cat"] = pd.Categorical(d.category, categories=CATS)
        for cat in CATS:
            dc = d[d.category == cat]
            if len(dc) < 10:
                e1_rows.append((g, cat, sample, np.nan, np.nan, len(dc)))
                continue
            f = "share_sent ~ beliefs_hp" + (" + C(story)" if sample == "Pooled" else "")
            m = smf.ols(f, data=dc).fit(cov_type="HC1")
            e1_rows.append((g, cat, sample, m.params["beliefs_hp"], m.bse["beliefs_hp"], int(m.nobs)))
        # joint model: equality of slopes across categories
        f = ('share_sent ~ beliefs_hp * C(cat, Treatment(reference="Moral"))'
             + (" + C(story)" if sample == "Pooled" else ""))
        mj = smf.ols(f, data=d).fit(cov_type="HC1")
        inter = [t for t in mj.params.index if t.startswith("beliefs_hp:")]
        eq = mj.f_test(" = 0, ".join(inter) + " = 0")
        eq_tests[(g, sample)] = float(eq.pvalue)
        log(f"{g.upper()} {sample}: slope-equality F-test p = {float(eq.pvalue):.4f}")

e1 = pd.DataFrame(e1_rows, columns=["game", "cat", "sample", "slope", "se", "n"])
for _, r in e1.iterrows():
    log(f"  {r.game.upper():3s} {r['sample']:8s} {CAT_SHORT[r['cat']]:22s} "
        f"slope={r.slope: .3f} (SE {r.se:.3f}, N={r.n})" if np.isfinite(r.slope)
        else f"  {r.game.upper():3s} {r['sample']:8s} {CAT_SHORT[r['cat']]:22s} N={r.n} (too small)")

# Figure: binned means + per-category fits, control sample
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, g in zip(axes, ["ug", "tg"]):
    d0 = p1[(p1.game == g) & (p1.story == 0)]
    d0 = d0[d0.classified & d0.beliefs_hp.notna() & d0.share_sent.notna()]
    for cat in CATS:
        dc = d0[d0.category == cat]
        n = len(dc)
        if n < 10:
            continue
        nb = int(np.clip(n // 40, 3, 8))
        q = pd.qcut(dc.beliefs_hp, nb, duplicates="drop")
        bx = dc.groupby(q, observed=True).beliefs_hp.mean()
        by = dc.groupby(q, observed=True).share_sent.mean()
        bn = dc.groupby(q, observed=True).size()
        ax.scatter(bx, by, s=18 + 3.5 * np.sqrt(bn), marker=CAT_MARKER[cat],
                   facecolor=CAT_COLOR[cat], edgecolor="black", linewidth=0.5, zorder=3)
        m = smf.ols("share_sent ~ beliefs_hp", data=dc).fit(cov_type="HC1")
        xs = np.linspace(dc.beliefs_hp.quantile(0.05), dc.beliefs_hp.quantile(0.95), 50)
        ax.plot(xs, m.params["Intercept"] + m.params["beliefs_hp"] * xs,
                color=CAT_COLOR[cat], linestyle=CAT_LS[cat], linewidth=2, zorder=2,
                label=f"{CAT_SHORT[cat]} (slope {m.params['beliefs_hp']:+.2f})")
    ax.set_title("Ultimatum Game" if g == "ug" else "Trust Game")
    ax.set_xlabel(BELIEF_LABEL[g])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    # Legends pinned flush to empty corners: the FONT_SCALE bump makes the default
    # upper-left (TG) collide with the MBC line and points (cf. the 07-15 lesson on
    # hand-tuned anchors).
    if g == "tg":
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.01))
    else:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01))
axes[0].set_ylabel("Action (share)")
fig.tight_layout()
fig.savefig(FIG_DIR / "player1_control_action_vs_belief_by_category.png", dpi=300)
plt.close(fig)
log("saved output/figures/player1_control_action_vs_belief_by_category.png")

# Table
def fmt(v, dec=3):
    return "--" if not np.isfinite(v) else f"{v:.{dec}f}"

lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.2}",
    r"\caption{\textbf{Belief Sensitivity of Actions, by Representation Category}}",
    r"\label{tab:belief_sensitivity}",
    r"\begin{tabular}{ll rr rr}", r"\toprule",
    r" & & \multicolumn{2}{c}{Control} & \multicolumn{2}{c}{All conditions (condition FE)} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"Game & Category & Slope & (SE) & Slope & (SE) \\", r"\midrule",
]
for g in ["ug", "tg"]:
    for cat in CATS:
        rc = e1[(e1.game == g) & (e1["cat"] == cat) & (e1["sample"] == "Control")].iloc[0]
        rp = e1[(e1.game == g) & (e1["cat"] == cat) & (e1["sample"] == "Pooled")].iloc[0]
        gname = GAME_LABELS[g] if cat == CATS[0] else ""
        lines.append(
            f"{gname} & {CAT_SHORT[cat]} & {fmt(rc.slope)} & ({fmt(rc.se)}) & "
            f"{fmt(rp.slope)} & ({fmt(rp.se)}) \\\\")
    if g == "ug":
        lines.append(r"\midrule")
lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: OLS slopes of the Player 1 action (share) on the hypothetical belief at "
    r"the one-third reference action, estimated separately by representation category; robust (HC1) "
    r"standard errors. ``All conditions'' pools the four conditions with condition fixed effects. "
    f"Slope-equality F-tests across categories: UG control $p={eq_tests[('ug', 'Control')]:.3f}$, "
    f"UG pooled $p={eq_tests[('ug', 'Pooled')]:.3f}$, TG control $p={eq_tests[('tg', 'Control')]:.3f}$, "
    f"TG pooled $p={eq_tests[('tg', 'Pooled')]:.3f}$. "
    r"Model predictions: negative slopes in the UG, positive in the TG for every category (the "
    r"equal-payoff norm anchor moves with the believed return); magnitude largest for "
    r"Self-interest, whose material channel adds to the common anchor channel. "
    r"The UG Mutual Benefit/Cooperation cell is very small and should be read accordingly.",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "belief_sensitivity_regs.tex").write_text("\n".join(lines) + "\n")
log("saved output/tables/belief_sensitivity_regs.tex")
log()

# ---------------------------------------------------------------------
# E2. Control vs Market distributions (DG-KW, UG, TG)
# ---------------------------------------------------------------------

log("=== E2: Control vs Market outcome distributions ===")
DOLLARS = {"dgkw": 12, "ug": 12, "tg": 6}
TITLES = {"dgkw": "Dictator (DG-KW)", "ug": "Ultimatum", "tg": "Trust"}

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
ymax = 0
panels = []
for g in ["dgkw", "ug", "tg"]:
    a = cell(g, 0)
    b = cell(g, 1)
    D = DOLLARS[g]
    av = (a * D).round().astype(int).value_counts(normalize=True).reindex(range(D + 1), fill_value=0)
    bv = (b * D).round().astype(int).value_counts(normalize=True).reindex(range(D + 1), fill_value=0)
    ks = stats.ks_2samp(a, b)
    log(f"{GAME_LABELS[g]}: N control={len(a)}, market={len(b)}; "
        f"mean {a.mean():.3f} vs {b.mean():.3f}; KS D={ks.statistic:.3f}, p={ks.pvalue:.2e}; "
        f"distinct raw values={a.nunique()}/{b.nunique()}")
    for arm_label, v in (("control", av), ("market ", bv)):
        log(f"  {GAME_LABELS[g]} {arm_label}: share at $0 = {v[0]:.3f}, "
            f"at equal split ${D // 2} = {v[D // 2]:.3f}, at max ${D} = {v[D]:.3f}")
    panels.append((g, av, bv))
    ymax = max(ymax, av.max(), bv.max())

for ax, (g, av, bv) in zip(axes, panels):
    D = DOLLARS[g]
    x = np.arange(D + 1)
    ax.bar(x - 0.21, av.values, width=0.42, color=CONTROL_COLOR, edgecolor="black",
           linewidth=0.4, label="Control", zorder=3)
    ax.bar(x + 0.21, bv.values, width=0.42, color=MARKET_COLOR, edgecolor="black",
           linewidth=0.4, label="Market", zorder=3)
    ax.axvline(D / 2, color="#444444", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_title(TITLES[g])
    ax.set_xlabel(f"Dollars sent (of \\${D})")
    ax.set_xticks(x[:: 2 if D == 12 else 1])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_ylim(0, ymax * 1.12)
axes[0].set_ylabel("Share of participants")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "market_vs_control_distributions.png", dpi=300)
plt.close(fig)
log("saved output/figures/market_vs_control_distributions.png")
log()

# ---------------------------------------------------------------------
# E3. Heterogeneity: between/within-category decomposition + predicted dispersion
# ---------------------------------------------------------------------

log("=== E3: heterogeneity decomposition (classified sample) ===")


def decomp(d):
    """Total var, between-category share (R^2 of action on category dummies), within var."""
    y = d.share_sent
    tot = y.var(ddof=1)
    grand = y.mean()
    groups = d.groupby("category").share_sent
    nb = sum(len(gr) * (gr.mean() - grand) ** 2 for _, gr in groups)
    between = nb / (len(y) - 1)
    return tot, between / tot, tot - between


rows = []
for g in GAMES:
    for s in [0, 1, 2, 4]:
        d = p1[(p1.game == g) & (p1.story == s) & p1.classified & p1.share_sent.notna()]
        tot, bshare, within = decomp(d)
        rows.append([GAME_LABELS[g], STORY[s], len(d), np.sqrt(tot), bshare])
        log(f"  {GAME_LABELS[g]:5s} {STORY[s]:8s} N={len(d):4d}  SD={np.sqrt(tot):.3f}  "
            f"between-category share={bshare:.3f}")
het = pd.DataFrame(rows, columns=["Game", "Arm", "N", "SD", "Between share"])

lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\caption{\textbf{Heterogeneity Tied to Representations: Variance Decomposition}}",
    r"\label{tab:heterogeneity_decomposition}",
    r"\begin{tabular}{ll rrr}", r"\toprule",
    r"Game & Condition & $N$ & SD of action & Between-category share \\", r"\midrule",
]
prev = None
for _, r in het.iterrows():
    gname = r.Game if r.Game != prev else ""
    prev = r.Game
    lines.append(f"{gname} & {r.Arm} & {r.N} & {r.SD:.3f} & {r['Between share']:.3f} \\\\")
lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Player 1, classified responses only. ``Between-category share'' is the "
    r"share of the within-condition variance of the action lying between the three representation "
    r"categories (the $R^2$ of the action on category dummies).",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "heterogeneity_decomposition.tex").write_text("\n".join(lines) + "\n")
log("saved output/tables/heterogeneity_decomposition.tex")

# Slimmed main-text version: the three control rows (DG-KW, UG, TG) plus the
# DG-KW Market row (the 0.66 -> 0.14 between-category collapse with the SD rise).
# Same booktabs layout as the full table; full decomposition (all games,
# conditions, DG-LT) stays in heterogeneity_decomposition.tex in the appendix.
main_selection = [("DG-KW", "Control"), ("DG-KW", "Market"),
                  ("UG", "Control"), ("TG", "Control")]
het_indexed = het.set_index(["Game", "Arm"])
main_lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\caption{\textbf{Heterogeneity Tied to Representations: Variance Decomposition}}",
    r"\label{tab:heterogeneity_decomposition_main}",
    r"\begin{tabular}{ll rrr}", r"\toprule",
    r"Game & Condition & $N$ & SD of action & Between-category share \\", r"\midrule",
]
prev = None
for game_name, arm_name in main_selection:
    r = het_indexed.loc[(game_name, arm_name)]
    gname = game_name if game_name != prev else ""
    prev = game_name
    main_lines.append(f"{gname} & {arm_name} & {int(r.N)} & {r.SD:.3f} & {r['Between share']:.3f} \\\\")
main_lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Player 1, classified responses only. ``Between-category share'' is the "
    r"share of the within-condition variance of the action lying between the three representation "
    r"categories (the $R^2$ of the action on category dummies). The full decomposition for all "
    r"games, conditions, and DG-LT is reported in the appendix table.",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "heterogeneity_decomposition_main.tex").write_text("\n".join(main_lines) + "\n")
log("saved output/tables/heterogeneity_decomposition_main.tex")

log("--- predicted vs actual dispersion (mixture reweighting, pooled conditionals) ---")
comp_rows = []
for base, treat, label in [(2, 4, "Aid vs Bonus"), (0, 1, "Market vs Control")]:
    for g in GAMES:
        d = p1[(p1.game == g) & p1.story.isin([base, treat]) & p1.classified & p1.share_sent.notna()]
        m_c = d.groupby("category").share_sent.mean()
        v_c = d.groupby("category").share_sent.var(ddof=1)

        def mixture(arm):
            da = d[d.story == arm]
            w = da.category.value_counts(normalize=True)
            mu = (w * m_c.reindex(w.index)).sum()
            var = (w * v_c.reindex(w.index)).sum() + (w * m_c.reindex(w.index) ** 2).sum() - mu ** 2
            return mu, var, da.share_sent.mean(), da.share_sent.var(ddof=1)

        mu_b, pv_b, am_b, av_b = mixture(base)
        mu_t, pv_t, am_t, av_t = mixture(treat)
        pred_dsd = np.sqrt(pv_t) - np.sqrt(pv_b)
        act_dsd = np.sqrt(av_t) - np.sqrt(av_b)
        pred_dm = mu_t - mu_b
        act_dm = am_t - am_b
        comp_rows.append([label, GAME_LABELS[g], 100 * pred_dm, 100 * act_dm, pred_dsd, act_dsd])
        log(f"  {label:18s} {GAME_LABELS[g]:5s} dMean pred={100*pred_dm:+6.2f}pp act={100*act_dm:+6.2f}pp | "
            f"dSD pred={pred_dsd:+.4f} act={act_dsd:+.4f}")

lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\caption{\textbf{Predicted versus Actual Treatment Effects on Dispersion}}",
    r"\label{tab:heterogeneity_predicted_actual}",
    r"\begin{tabular}{ll rr rr}", r"\toprule",
    r" & & \multicolumn{2}{c}{$\Delta$ Mean (pp)} & \multicolumn{2}{c}{$\Delta$ SD} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"Comparison & Game & Predicted & Actual & Predicted & Actual \\", r"\midrule",
]
prev = None
for label, g, pdm, adm, psd, asd in comp_rows:
    cname = label if label != prev else ""
    prev = label
    lines.append(f"{cname} & {g} & {pdm:+.1f} & {adm:+.1f} & {psd:+.3f} & {asd:+.3f} \\\\")
    if g == "TG" and label == "Aid vs Bonus":
        lines.append(r"\midrule")
lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Player 1, classified responses only. Predicted values reweight the "
    r"category-conditional action distributions (means and variances pooled across the two conditions of "
    r"each comparison) by each condition's category shares; the predicted effect is the mixture "
    r"implication of the representation shift alone. Actual values are the raw between-condition differences.",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "heterogeneity_predicted_actual.tex").write_text("\n".join(lines) + "\n")
log("saved output/tables/heterogeneity_predicted_actual.tex")

# Figure version of the dispersion accounting: predicted vs actual on the mean
# and the SD, six cells only (DG-KW/UG/TG; DG-LT stays in the appendix table).
# Style mirrors the predicted-vs-actual scatters in output/figures/fitted_fullpooled/
# (white background, serif, 45-degree dashed reference line, light grid).
LABEL_TO_CODE = {v: k for k, v in GAME_LABELS.items()}
FIG_GAMES = ["dgkw", "ug", "tg"]
fig_rows = [r for r in comp_rows if LABEL_TO_CODE[r[1]] in FIG_GAMES]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
panel_specs = [
    (axes[0], r"$\Delta$ Mean of action", [(r[2], r[3]) for r in fig_rows]),
    (axes[1], r"$\Delta$ SD of action", [(100 * r[4], 100 * r[5]) for r in fig_rows]),
]
for ax, panel_title, pts in panel_specs:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo = min(xs + ys)
    hi = max(xs + ys)
    span = hi - lo if hi > lo else 1.0
    lo, hi = lo - 0.12 * span, hi + 0.12 * span
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1, zorder=1)
    for r, (px, py) in zip(fig_rows, pts):
        ax.scatter(
            px, py,
            s=60,
            color=GAME_COLOR[LABEL_TO_CODE[r[1]]],
            marker=COMPARISON_MARKER[r[0]],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(panel_title)
    ax.set_xlabel("Predicted (pp of budget share)")
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Actual (pp of budget share)")

game_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=GAME_COLOR[g], markeredgecolor="black",
               markeredgewidth=0.5, label=GAME_LABELS[g])
    for g in FIG_GAMES
]
comparison_handles = [
    Line2D([], [], marker=COMPARISON_MARKER[label], linestyle="none", markersize=9,
               markerfacecolor="#BBBBBB", markeredgecolor="black",
               markeredgewidth=0.5, label=label)
    for label in ["Market vs Control", "Aid vs Bonus"]
]
fig.legend(
    handles=game_handles + comparison_handles,
    frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG_DIR / "heterogeneity_predicted_actual.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log("saved output/figures/heterogeneity_predicted_actual.png")
log()

# ---------------------------------------------------------------------
# E4. KW-LT gap on the treatment scale
# ---------------------------------------------------------------------

log("=== E4: KW-LT framing gap vs treatment effects (DG, share of $12 budget) ===")
kw0, lt0 = cell("dgkw", 0), cell("dglt", 0)
gap = (lt0.mean() - kw0.mean()) * 100
boots = np.array([
    (rng.choice(lt0, len(lt0)).mean() - rng.choice(kw0, len(kw0)).mean()) * 100
    for _ in range(5000)
])
gap_ci = np.percentile(boots, [2.5, 97.5])
check("KW-LT gap (pp)", gap, 6.4, 0.3)

ab_d, ab_se = welch_diff(cell("dgkw", 2), cell("dgkw", 4))
mc_d, mc_se = welch_diff(cell("dgkw", 0), cell("dgkw", 1))
check("Aid-Bonus effect on DG-KW (pp)", 100 * ab_d, -5.6, 0.3)
check("Market-Control effect on DG-KW (pp)", 100 * mc_d, -10.9, 0.3)

rows_e4 = [
    ("KW--LT framing gap (LT $-$ KW, control)", gap, gap_ci[0], gap_ci[1]),
    ("Aid vs Bonus (DG-KW)", 100 * ab_d, 100 * (ab_d - 1.96 * ab_se), 100 * (ab_d + 1.96 * ab_se)),
    ("Market vs Control (DG-KW)", 100 * mc_d, 100 * (mc_d - 1.96 * mc_se), 100 * (mc_d + 1.96 * mc_se)),
]
for lab, d, lo, hi in rows_e4:
    log(f"  {lab}: {d:+.1f}pp  [95% CI {lo:+.1f}, {hi:+.1f}]")

lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\caption{\textbf{The KW--LT Gap on the Treatment Scale (Dictator Game)}}",
    r"\label{tab:kwlt_treatment_scale}",
    r"\begin{tabular}{l rc}", r"\toprule",
    r"Contrast & Effect (pp of budget) & 95\% CI \\", r"\midrule",
]
for lab, d, lo, hi in rows_e4:
    lines.append(f"{lab} & ${d:+.1f}$ & $[{lo:+.1f},\\,{hi:+.1f}]$ \\\\")
lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Player 1 mean transfers as shares of the \$12 budget, in percentage "
    r"points. The framing gap compares the two payoff-equivalent dictator games in the control "
    r"condition (bootstrap CI, 5{,}000 draws); the treatment rows are randomized contrasts within "
    r"DG-KW (Welch CIs).",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "kwlt_treatment_scale.tex").write_text("\n".join(lines) + "\n")
log("saved output/tables/kwlt_treatment_scale.tex")
log()

# ---------------------------------------------------------------------
# E5. Similarity gradient vs story effects
# ---------------------------------------------------------------------

log("=== E5: similarity gradient vs Aid-Bonus effects ===")
SIM = {"dgkw": 94.6, "dglt": 51.2, "ug": 29.9, "tg": 14.4}      # audited, LLM workbook
RETR = {"dgkw": 41, "dglt": 54, "ug": 60, "tg": 72}             # Bonus retrieval share

e5_rows = []
for g in GAMES:
    b, a = cell(g, 2), cell(g, 4)
    d_act, se_act = welch_diff(b, a)
    bm = p1[(p1.game == g) & (p1.story == 2) & p1.classified]
    am = p1[(p1.game == g) & (p1.story == 4) & p1.classified]
    d_moral = (am.category == "Moral").mean() - (bm.category == "Moral").mean()
    e5_rows.append([GAME_LABELS[g], SIM[g], RETR[g], 100 * d_act, 100 * 1.96 * se_act, 100 * d_moral])
    log(f"  {GAME_LABELS[g]:5s} similarity={SIM[g]:5.1f}  aid-bonus action={100*d_act:+.1f}pp "
        f"(±{100*1.96*se_act:.1f})  aid-bonus Moral share={100*d_moral:+.1f}pp")

check("Aid-Bonus on UG offers (pp)", e5_rows[2][3], -2.4, 0.3)
check("Aid-Bonus on TG sends (pp)", e5_rows[3][3], 0.5, 0.3)
check("Aid-Bonus on DG-KW Moral share (pp)", e5_rows[0][5], -11.7, 0.5)
check("Aid-Bonus on UG Moral share (pp)", e5_rows[2][5], -9.7, 0.5)

abs_eff = [abs(r[3]) for r in e5_rows]
sims = [r[1] for r in e5_rows]
rho = stats.spearmanr(sims, abs_eff)
log(f"  Spearman(|action effect|, similarity) over 4 games: rho={rho.statistic:.2f} (descriptive)")

lines = [
    r"\begin{table}[!htbp]", r"\centering", r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\caption{\textbf{Story Effects Track the Structural Similarity of Stories to Games}}",
    r"\label{tab:similarity_gradient}",
    r"\begin{tabular}{l cc rr}", r"\toprule",
    r" & Structural & Bonus retrieval & \multicolumn{2}{c}{Aid $-$ Bonus effect (pp)} \\",
    r"\cmidrule(lr){4-5}",
    r"Game & similarity (0--100) & share (\%) & Action & Moral share \\", r"\midrule",
]
for gl, sim, retr, dact, ci, dmor in e5_rows:
    lines.append(f"{gl} & {sim:.1f} & {retr} & ${dact:+.1f}$ & ${dmor:+.1f}$ \\\\")
lines += [
    r"\bottomrule", r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Structural similarity and retrieval shares are means over the nine LLM "
    r"conversations (Section on Treatments and Similarity); action and Moral-share effects are "
    r"Aid $-$ Bonus differences (Player 1; Moral share among classified responses). The "
    r"treatment-effect figures of Section 4 use DG-KW, UG, TG\@; the DG-LT story conditions enter "
    r"this table and the dispersion accounting of Section 5.",
    r"\end{flushleft}", r"\end{table}",
]
(TABLE_DIR / "similarity_gradient.tex").write_text("\n".join(lines) + "\n")
log("saved output/tables/similarity_gradient.tex")
log()

STATS_FILE.write_text("\n".join(stats_lines) + "\n")
log(f"stats log written to {STATS_FILE.relative_to(ROOT)}")
