"""
04_paper_v1_extra_outputs.py

New outputs for paper_v1.tex (working paper version 1.0):

1. Intro figure: distribution of Player 1 allocations in DG-KW vs DG-LT,
   control condition (no story, abstract frame), between-subject sample.
2. Discussion table: treatment effects on realized surplus in UG and TG.
3. Appendix table: within-subject switching summary (choices, categories,
   social proximity) from within/within_switching_results.xlsx.

Existing tables and figures are NOT touched; all outputs are new files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # replication_package/
P1_FILE = ROOT / "data" / "player1_all_categorized.xlsx"
P2_FILE = ROOT / "data" / "player2_all_categorized.xlsx"
WITHIN_FILE = ROOT / "data" / "within_switching_results.xlsx"
FIG_DIR = ROOT / "output" / "figures"
TABLE_DIR = ROOT / "output" / "tables"
STATS_FILE = ROOT / "output" / "tables" / "paper_v1_extra_stats.txt"

# Drawn ~8.5in wide, included at ~5.9in: scale the fonts so they land near 9pt on
# the page rather than being optically shrunk to ~7pt. See FONT_SCALE in script 01.
FONT_SCALE = 1.25

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
    }
)

KW_COLOR = "#4C566A"
LT_COLOR = "#C65D3A"

stats_lines = []


def log(line=""):
    print(line)
    stats_lines.append(line)


# ---------------------------------------------------------------------
# 1. Intro figure: DG-KW vs DG-LT control allocation distributions
# ---------------------------------------------------------------------

p1 = pd.read_excel(P1_FILE)
kw = p1.loc[(p1.game == "dgkw") & (p1.story == 0), "share_sent"].dropna()
lt = p1.loc[(p1.game == "dglt") & (p1.story == 0), "share_sent"].dropna()

# Bin at the twelve salient dollar values (share of the $12 budget).
kw_bins = (kw * 12).round().astype(int).clip(0, 12)
lt_bins = (lt * 12).round().astype(int).clip(0, 12)

x = np.arange(13)
kw_pct = np.array([(kw_bins == v).mean() * 100 for v in x])
lt_pct = np.array([(lt_bins == v).mean() * 100 for v in x])

fig, ax = plt.subplots(figsize=(8.5, 5.0))
width = 0.42
ax.bar(x - width / 2, kw_pct, width, color=KW_COLOR, label=r"DG-KW (initial allocation \$12 / \$0)", zorder=3)
ax.bar(x + width / 2, lt_pct, width, color=LT_COLOR, label=r"DG-LT (initial allocation \$8 / \$4)", zorder=3)

ax.axvline(0, color=KW_COLOR, linestyle=":", linewidth=1.4, alpha=0.6, zorder=1)
ax.axvline(4, color=LT_COLOR, linestyle=":", linewidth=1.4, alpha=0.6, zorder=1)
ax.annotate("DG-KW default", xy=(0.3, 24), fontsize=9, color=KW_COLOR)
ax.annotate("DG-LT default", xy=(4.3, 24), fontsize=9, color=LT_COLOR)

ax.set_xticks(x)
ax.set_xticklabels([f"{v}" for v in x])
ax.set_xlabel(r"Dollars to the other participant (final allocation, out of \$12)")
ax.set_ylabel("Share of participants (%)")
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG_DIR / "intro_kw_vs_lt_control.png", dpi=300)
plt.close(fig)

mw = stats.mannwhitneyu(kw, lt, alternative="two-sided")
ks = stats.ks_2samp(kw, lt)
tt = stats.ttest_ind(kw, lt, equal_var=False)

log("=== Intro figure: DG-KW vs DG-LT, control (between-subject P1) ===")
log(f"N: KW={len(kw)}, LT={len(lt)}")
log(f"Mean share to recipient: KW={kw.mean():.3f}, LT={lt.mean():.3f}")
log(f"Mean dollars to recipient: KW={12 * kw.mean():.2f}, LT={12 * lt.mean():.2f}")
log(f"Share at $0 to recipient: KW={(kw_bins == 0).mean():.3f}, LT={(lt_bins == 0).mean():.3f}")
log(f"Share at default: KW($0)={(kw_bins == 0).mean():.3f}, LT($4)={(lt_bins == 4).mean():.3f}")
log(f"Share at equal split ($6): KW={(kw_bins == 6).mean():.3f}, LT={(lt_bins == 6).mean():.3f}")
log(f"Share giving MORE than half: KW={(kw_bins > 6).mean():.3f}, LT={(lt_bins > 6).mean():.3f}")
log(f"Welch t-test on means: t={tt.statistic:.2f}, p={tt.pvalue:.4f}")
log(f"Mann-Whitney: p={mw.pvalue:.4f}; KS test: D={ks.statistic:.3f}, p={ks.pvalue:.5f}")
log()

# ---------------------------------------------------------------------
# 2. Treatment effects on realized surplus (UG and TG)
# ---------------------------------------------------------------------

p2 = pd.read_excel(P2_FILE)
STORY = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}

# UG: surplus = $12 if the receiver accepts, $0 otherwise (per matched pair).
ug = p2[(p2.game == "ug") & p2.story.isin(STORY) & p2.choice.notna()].copy()
ug["story"] = ug["story"].astype(int)

# TG: surplus = 6*(1-s) + 18*s = 6 + 12s dollars; determined by P1's send.
tg = p1[(p1.game == "tg") & p1.story.isin(STORY) & p1.share_sent.notna()].copy()
tg["surplus"] = 6 + 12 * tg["share_sent"]


def cell(series):
    return series.mean(), series.std(ddof=1) / np.sqrt(len(series)), len(series)


def diff_test(a, b, proportion=False):
    """b minus a, with Welch t-test p-value."""
    res = stats.ttest_ind(b, a, equal_var=False)
    return b.mean() - a.mean(), res.pvalue


rows = []
log("=== Realized surplus by treatment ===")
for label, frame, var, scale in [
    ("UG: acceptance rate", ug, "choice", 1),
    (r"UG: expected surplus (\$)", ug, "choice", 12),
    (r"TG: realized surplus (\$)", tg, "surplus", 1),
]:
    vals = {}
    for s, name in STORY.items():
        v = frame.loc[frame.story == s, var] * scale
        vals[name] = v
        m, se, n = cell(v)
        log(f"{label} | {name}: mean={m:.3f} (se={se:.3f}, n={n})")
    d_mc, p_mc = diff_test(vals["Control"], vals["Market"])
    d_ab, p_ab = diff_test(vals["Bonus"], vals["Aid"])
    log(f"{label} | Market-Control: diff={d_mc:.3f} (p={p_mc:.4f}); Aid-Bonus: diff={d_ab:.3f} (p={p_ab:.4f})")
    rows.append((label, vals, d_mc, p_mc, d_ab, p_ab))
log()


def starp(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""


def fmt(label, v):
    dec = 3 if "rate" in label else 2
    return f"{v:.{dec}f}"


lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{\textbf{Treatment Effects on Realized Surplus}}",
    r"\label{tab:surplus_by_treatment}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r" & Control & Market & Market $-$ Control & Bonus & Aid & Aid $-$ Bonus \\",
    r"\midrule",
]
for label, vals, d_mc, p_mc, d_ab, p_ab in rows:
    lines.append(
        f"{label} & {fmt(label, vals['Control'].mean())} & {fmt(label, vals['Market'].mean())} & "
        f"{fmt(label, d_mc)}{starp(p_mc)} & {fmt(label, vals['Bonus'].mean())} & {fmt(label, vals['Aid'].mean())} & "
        f"{fmt(label, d_ab)}{starp(p_ab)} \\\\ % chktex 13"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: In the Ultimatum Game, realized surplus per matched pair is \$12 if the receiver accepts and \$0 otherwise; the acceptance rate is computed over receivers responding to their matched sender's offer. In the Trust Game, realized surplus is $6+12s$ dollars, where $s$ is the sender's share sent; the receiver's return does not change total surplus. The Dictator Game is omitted because total surplus is fixed at \$12 by design. Differences are tested with Welch two-sample $t$-tests. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
    r"\end{flushleft}",
    r"\end{table}",
]
(TABLE_DIR / "surplus_by_treatment.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# 3. Within-subject switching summary table
# ---------------------------------------------------------------------

summ = pd.read_excel(WITHIN_FILE, sheet_name="summary_all")
keep = summ[
    ((summ.design == "control_kwlt") & (summ.fixed_game == "pooled"))
    | ((summ.design == "market_control") & (summ.fixed_game.isin(["kw", "lt"])))
].copy()

row_labels = {
    ("control_kwlt", "pooled"): r"DG-KW vs DG-LT (abstract frame)",
    ("market_control", "kw"): r"Market vs Control (DG-KW)",
    ("market_control", "lt"): r"Market vs Control (DG-LT)",
}

lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{\textbf{Within-Subject Switching in the Dictator Game}}",
    r"\label{tab:within_switching_summary}",
    r"\resizebox{0.98\textwidth}{!}{%",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r" & & \multicolumn{3}{c}{Share switching} & \multicolumn{2}{c}{P(choice switch $\mid$ category)} \\",
    r"\cmidrule(lr){3-5} \cmidrule(lr){6-7}",
    r"Comparison & $N$ pairs & Choice & Category & Social proximity & Switched & Stayed \\",
    r"\midrule",
]
for _, r in keep.iterrows():
    lab = row_labels[(r.design, r.fixed_game)]
    lines.append(
        f"{lab} & {int(r.n_pairs)} & {r.share_choice_switched * 100:.1f}\\% & "
        f"{r.share_category_switched * 100:.1f}\\% & {r.share_sp_switched * 100:.1f}\\% & "
        f"{r.choice_switch_rate_if_category_switches * 100:.1f}\\% & "
        f"{r.choice_switch_rate_if_category_stays * 100:.1f}\\% \\\\"
    )
    log(
        f"WITHIN {lab}: N={int(r.n_pairs)}, choice switch={r.share_choice_switched:.3f}, "
        f"category switch={r.share_category_switched:.3f}, sp switch={r.share_sp_switched:.3f}, "
        f"P(choice|cat switch)={r.choice_switch_rate_if_category_switches:.3f}, "
        f"P(choice|cat stay)={r.choice_switch_rate_if_category_stays:.3f}"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\begin{flushleft}",
    r"\footnotesize Notes: Each participant makes two dictator-game decisions. In the first row, both decisions use the abstract (control) frame and the game varies (KW vs LT); in the second and third rows, the game is fixed and the frame varies (abstract vs market). ``Choice'' switching is any change in the share sent; ``Category'' switching is a change in the reason-based representation category (Moral, Self-interest, Mutual Benefit/Cooperation, No clear justification); ``Social proximity'' switching is a change in the social-proximity class of the representation. The last two columns report the probability of a choice switch conditional on the representation category switching or staying fixed.",
    r"\end{flushleft}",
    r"\end{table}",
]
(TABLE_DIR / "within_switch_summary.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# 4. Additional inference requested by internal review
# ---------------------------------------------------------------------

# (a) LPM of choice switching on category switching, per within-subject arm.
import statsmodels.formula.api as smf

pairs = pd.read_excel(WITHIN_FILE, sheet_name="pair_level_data")
log("=== LPM: choice switch on category switch (robust HC1 SEs) ===")
arms = [
    ("control_kwlt", "pooled", "DG-KW vs DG-LT (abstract frame)"),
    ("market_control", "kw", "Market vs Control (DG-KW)"),
    ("market_control", "lt", "Market vs Control (DG-LT)"),
]
for design, fixed, lab in arms:
    sub = pairs[(pairs.design == design) & (pairs.fixed_game == fixed)]
    if sub.empty and fixed == "pooled":
        sub = pairs[pairs.design == design]
    m = smf.ols("choice_switched ~ category_switched", data=sub).fit(cov_type="HC1")
    log(
        f"{lab}: N={int(m.nobs)}, b={m.params['category_switched']:.3f} "
        f"(robust SE {m.bse['category_switched']:.3f}, p={m.pvalues['category_switched']:.2g})"
    )
log()

# (b) 95% CIs for the two headline Market-vs-Control treatment effects.
log("=== Welch 95% CIs for headline treatment effects (pp of budget) ===")
for g, lab in [("tg", "TG share sent"), ("ug", "UG share offered")]:
    a = p1.loc[(p1.game == g) & (p1.story == 0), "share_sent"].dropna()
    b = p1.loc[(p1.game == g) & (p1.story == 1), "share_sent"].dropna()
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    log(f"Market-Control, {lab}: {100*diff:.1f} pp, 95% CI [{100*(diff-1.96*se):.1f}, {100*(diff+1.96*se):.1f}]")

# ---------------------------------------------------------------------
# 5. Within-subject choice-switching bubble plot (KW first, LT second)
# ---------------------------------------------------------------------
# Regenerated 2026-07-15. The original within/choice_switch_control_kw_first.png
# had no generator in the repo and was in a sans-serif face unlike every other
# figure in the paper. Filter and the >=3% label rule were recovered by matching
# the percentages printed on the original: all seven agree exactly (45.3/6.3/
# 5.3/3.7/3.3/3.3/3.0 on N=300). The original file is left in place, untouched.

switch = pairs[(pairs.design == "control_kwlt") & (pairs.order_label == "KW first")].dropna(
    subset=["share_sent_kw", "share_sent_lt"]
)
n_switch = len(switch)
cells = switch.groupby(["share_sent_kw", "share_sent_lt"]).size().reset_index(name="n")
cells["pct"] = 100 * cells["n"] / n_switch

# Marker area is proportional to the number of participants in the cell, so the
# bubble encodes frequency directly; AREA_PER_PAIR sets the largest (n=136) bubble
# to about a tenth of the axis range, as in the original.
AREA_PER_PAIR = 11.0

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.2, zorder=1)
ax.scatter(
    cells["share_sent_kw"],
    cells["share_sent_lt"],
    s=cells["n"] * AREA_PER_PAIR,
    facecolor=KW_COLOR,
    edgecolor="#2E3440",
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
)
for r in cells[cells["pct"] >= 3].itertuples():
    # Offset diagonally by the marker's own radius: clears the bubble, and going
    # up-and-right also dodges the column of bubbles sharing the same KW share.
    radius_pt = np.sqrt(r.n * AREA_PER_PAIR / np.pi)
    ax.annotate(
        f"{r.pct:.1f}%",
        (r.share_sent_kw, r.share_sent_lt),
        xytext=(0.8 * radius_pt + 4, 0.8 * radius_pt + 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        zorder=4,
    )

ticks = [0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([f"{t:.3f}" for t in ticks])
ax.set_yticklabels([f"{t:.3f}" for t in ticks])
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-0.06, 1.06)
ax.set_aspect("equal")
ax.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("KW share sent (first)")
ax.set_ylabel("LT share sent (second)")
ax.set_title("Choice Switching: KW first, LT second")
fig.tight_layout()
fig.savefig(FIG_DIR / "within_choice_switch_kw_first.png", dpi=300, bbox_inches="tight")
plt.close(fig)

log("=== Within-subject choice switching (control_kwlt, KW first) ===")
log(f"N pairs={n_switch}; cells labelled (>=3%)={int((cells['pct'] >= 3).sum())}")
for r in cells.sort_values("pct", ascending=False).head(7).itertuples():
    log(f"  KW={r.share_sent_kw:.4f} LT={r.share_sent_lt:.4f}: n={r.n}, {r.pct:.1f}%")
log()

STATS_FILE.write_text("\n".join(stats_lines) + "\n")
print("\nDone. Outputs: figures/intro_kw_vs_lt_control.png, figures/within_choice_switch_kw_first.png, paper_tables/surplus_by_treatment.tex, paper_tables/within_switch_summary.tex, paper_tables/paper_v1_extra_stats.txt")
