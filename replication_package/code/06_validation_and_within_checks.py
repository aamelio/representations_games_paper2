"""
06_validation_and_within_checks.py

Analyses unlocked by AA's 2026-07-10 delivery (eight workbooks, formerly in
data/missingdata/, moved into data/ on 2026-07-21):

A. LLM-human classification validation (six workbooks: player1 aid/bonus,
   player1 market/control, player2). The two 'subsamples' per batch are
   DISJOINT sets of responses (zero shared texts) drawn stratified on
   the LLM category (identical LLM label marginals by construction; 100
   responses per game-arm cell for Player 1; stratified draw CONFIRMED by
   AA 2026-07-10). Protocol (AA, 2026-07-10): classifier = GPT-4.1 prompted
   with the written classification rules; two RA coders, one per subsample
   throughout, blind to treatment and game. Outputs: agreement rate and
   Cohen's kappa by batch, subsample, and arm; consistency check against
   the analysis files (data/*_all_categorized.xlsx) matched on PID +
   response text (AA's merge key) --- labels IDENTICAL. The 1.5-3.9%
   'label drift' previously flagged was an artifact of merging on PID
   alone: repeat Prolific workers have multiple rows (different games/
   arms), so PID-only matching misaligns labels. RESOLVED 2026-07-10.
   Table: paper_tables/llm_human_agreement.tex.

B. Within-subject microdata checks (within_all_long/pairs_categorized.xlsx):
   arm Ns, switching rates, and LPM coefficients validated against the
   audited v1.1 numbers; full verification of AA's within regression tables.
   Result: every coefficient identified under participant fixed effects
   (KW/Market, order interactions, HighSP, all category dummies, R2, N)
   reproduces to the third decimal under strict FE (LSDV) with clustering.
   The LTFirst=0.104 (0.008) and ControlFirst=0.184 (0.013) rows are NOT
   identified under FE: AA confirmed (2026-07-10) they came from
   statsmodels' generalized (pseudo-inverse) solution to the rank-
   deficient design and must not be interpreted; order effects require a
   no-FE spec with participant-clustered SEs. Those rows are DROPPED from
   within/within_regression_table_{control,market}.tex; the no-FE
   clustered estimates (+0.019 (0.015) / +0.009 (0.013), printed below)
   now appear in the table notes. Identified order effects are SMALL
   (cell means printed below); the draft's 'order effects are sizable'
   parenthetical was corrected accordingly.

Prints aggregate statistics only; free-text response columns are never printed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent  # replication_package/
DATA = ROOT / "data"
TABLE_DIR = ROOT / "output" / "tables"
STATS_FILE = TABLE_DIR / "validation_within_stats.txt"

LINES = []


def emit(msg=""):
    print(msg)
    LINES.append(str(msg))


def agree_kappa(a, b):
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    keep = a.notna() & b.notna()
    a, b = a[keep].astype(int), b[keep].astype(int)
    po = (a == b).mean()
    labels = sorted(set(a) | set(b))
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return po, (po - pe) / (1 - pe), len(a)


# ----------------------------------------------------------------------------
# Part A: LLM-human validation
# ----------------------------------------------------------------------------

STORY = {0: "Control", 1: "Market", 2: "Bonus", 4: "Aid"}
P1_LABELS = {0: "No clear justification", 1: "Moral", 2: "Mutual Benefit / Cooperation", 3: "Self-interest"}
P2_LABELS = {0: "No clear justification", 1: "Moral good", 2: "Moral bad", 3: "Mutual Benefit / Cooperation", 4: "Self-interest"}

BATCHES = {
    "Player 1, story conditions (Aid/Bonus)": "player1_aid_bonus_reason_subsample_{}_compiled_with_llm_story_game.xlsx",
    "Player 1, Market/Control": "player1_market_control_reason_subsample_{}_compiled_with_llm_story_game.xlsx",
    "Player 2, all conditions": "player2_reason_subsample_{}_compiled_with_llm_story_game.xlsx",
}

emit("=" * 78)
emit("PART A: LLM-HUMAN CLASSIFICATION VALIDATION")
emit("=" * 78)

p1_all = pd.read_excel(ROOT / "data" / "player1_all_categorized.xlsx")
p2_all = pd.read_excel(ROOT / "data" / "player2_all_categorized.xlsx")
for _ref in (p1_all, p2_all):
    _ref["key_text"] = _ref["reasons"].astype(str).str.strip()

# sanity checks: label maps; disjointness of the two subsamples
for batch, pat in BATCHES.items():
    f1 = pd.read_excel(DATA / pat.format(1))
    f2 = pd.read_excel(DATA / pat.format(2))
    labels = P2_LABELS if batch.startswith("Player 2") else P1_LABELS
    m = f1.groupby("llm_category_num")["llm_category"].agg(lambda s: s.unique().tolist())
    for num, labs in m.items():
        assert labs == [labels[num]], f"label map mismatch in {batch}: {num} -> {labs}"
    shared_texts = set(f1["reason"].astype(str).str.strip()) & set(f2["reason"].astype(str).str.strip())
    assert not shared_texts, f"{batch}: {len(shared_texts)} response texts appear in both subsamples"
    shared_pids = set(f1["PROLIFIC_PID"]) & set(f2["PROLIFIC_PID"])
    if shared_pids:
        emit(f"[note] {batch}: {len(shared_pids)} PID(s) in both subsamples, but with different "
             f"response texts (repeat Prolific workers with rows in multiple games/arms; benign)")
emit("[OK] numeric label maps consistent; no response text was coded in both subsamples")
emit("     (identical LLM-label marginals across disjoint subsamples => stratified draw on the LLM category)")

arm_rows = {}     # batch -> list of (arm, (n1, po1, k1), (n2, po2, k2))
batch_rows = {}   # batch -> ((n1, po1, k1), (n2, po2, k2), pooled (n, po, k))
pooled_a, pooled_b = [], []

for batch, pat in BATCHES.items():
    emit(f"\n--- {batch} ---")
    per_sub, per_arm = [], {}
    all_a, all_b = [], []
    for sub in (1, 2):
        df = pd.read_excel(DATA / pat.format(sub)).dropna(subset=["cat_manual"])
        po, kap, n = agree_kappa(df["cat_manual"], df["llm_category_num"])
        emit(f"subsample {sub}: N={n}  agreement={po:.3f}  kappa={kap:.3f}")
        per_sub.append((n, po, kap))
        all_a.append(df["cat_manual"]); all_b.append(df["llm_category_num"])
        pooled_a.append(df["cat_manual"]); pooled_b.append(df["llm_category_num"])
        for arm, grp in df.groupby("story"):
            po_a, kap_a, n_a = agree_kappa(grp["cat_manual"], grp["llm_category_num"])
            emit(f"   arm {STORY[arm]:<8} N={n_a:>4}  agreement={po_a:.3f}  kappa={kap_a:.3f}")
            per_arm.setdefault(STORY[arm], {})[sub] = (n_a, po_a, kap_a)
        for game, grp in df.groupby("game"):
            po_g, kap_g, n_g = agree_kappa(grp["cat_manual"], grp["llm_category_num"])
            emit(f"   game {game:<7} N={n_g:>4}  agreement={po_g:.3f}  kappa={kap_g:.3f}")

        # consistency vs the analysis files, matched on PID + response text (AA's
        # merge key). PID-only matching is WRONG here: repeat Prolific workers have
        # multiple rows (different games/arms), which misaligns labels and
        # manufactured the spurious 1.5-3.9% 'drift' flagged before 2026-07-10.
        ref = p2_all if batch.startswith("Player 2") else p1_all
        dfk = df.copy()
        dfk["key_text"] = dfk["reason"].astype(str).str.strip()
        m = dfk.merge(ref.drop_duplicates(["PROLIFIC_PID", "key_text"]),
                      on=["PROLIFIC_PID", "key_text"], how="left")
        unmatched = int(m["category"].isna().sum())
        mm = m.dropna(subset=["category"])
        mismatch = int((mm["llm_category"] != mm["category"]).sum())
        emit(f"   [consistency] vs analysis-file labels (PID+text merge): matched={len(mm)}, "
             f"unmatched={unmatched}, label mismatches={mismatch}")

    po, kap, n = agree_kappa(pd.concat(all_a), pd.concat(all_b))
    emit(f"batch pooled: N={n}  agreement={po:.3f}  kappa={kap:.3f}")
    batch_rows[batch] = (per_sub[0], per_sub[1], (n, po, kap))
    arm_rows[batch] = per_arm

    both = pd.concat([pd.read_excel(DATA / pat.format(s)).dropna(subset=["cat_manual"]) for s in (1, 2)])
    both["cat_manual"] = both["cat_manual"].astype(int)
    dis = both[both["cat_manual"] != both["llm_category_num"]]
    labels = P2_LABELS if batch.startswith("Player 2") else P1_LABELS
    top = dis.groupby(["cat_manual", "llm_category_num"]).size().sort_values(ascending=False).head(2)
    for (hh, ll), cnt in top.items():
        emit(f"   top confusion: human={labels[hh]!r} vs LLM={labels[ll]!r}: {cnt} ({cnt/len(both):.1%})")

po_all, kap_all, n_all = agree_kappa(pd.concat(pooled_a), pd.concat(pooled_b))
emit(f"\nOVERALL: N={n_all}  agreement={po_all:.3f}  kappa={kap_all:.3f}")

# LaTeX table: batch x arm rows, subsample column groups
def fmt(cell):
    n, po, kap = cell
    return f"{n} & {po:.2f} & {kap:.2f}"

rows = []
for batch in BATCHES:
    rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{batch}}}}} \\")
    for arm in ("Control", "Market", "Bonus", "Aid"):
        if arm in arm_rows[batch]:
            c1 = fmt(arm_rows[batch][arm][1])
            c2 = fmt(arm_rows[batch][arm][2])
            rows.append(rf"\quad {arm} & {c1} & {c2} \\")
    s1, s2, _ = batch_rows[batch]
    rows.append(rf"\quad All conditions & {fmt(s1)} & {fmt(s2)} \\")
    rows.append(r"\addlinespace")
rows.append(rf"\emph{{Overall}} & \multicolumn{{6}}{{c}}{{$N={n_all:,}$, agreement $={po_all:.2f}$, $\kappa={kap_all:.2f}$}} \\")

table = rf"""\begin{{table}}[!htbp]
\centering
\caption{{\textbf{{Validation of the LLM Classification Against Human Coders}}}}
\label{{tab:llm_human_agreement}}
\begin{{tabular}}{{lcccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{Subsample 1}} & \multicolumn{{3}}{{c}}{{Subsample 2}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
 & $N$ & Agreement & $\kappa$ & $N$ & Agreement & $\kappa$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\begin{{flushleft}}
\footnotesize Notes: Human coders classified free-text answers to the reasons question using the same classification rules given to the LLM (GPT-4.1). Each batch comprises two disjoint validation subsamples drawn stratified on the LLM-assigned category (for Player 1, 100 responses per game--condition cell across DG-KW, UG, and TG\@; for Player 2, roughly 100 per game--condition cell across UG and TG in all four conditions); each subsample was coded by one of two research assistants, blind to the condition and to the game. Agreement is the share of responses assigned to the same category; $\kappa$ is Cohen's kappa over the full category set (four categories for Player 1, five for Player 2), including the residual ``No clear justification''.
\end{{flushleft}}
\end{{table}}
"""

# ----------------------------------------------------------------------------
# Part B: within-subject microdata
# ----------------------------------------------------------------------------

emit("\n" + "=" * 78)
emit("PART B: WITHIN-SUBJECT MICRODATA")
emit("=" * 78)

long = pd.read_excel(DATA / "within_all_long_categorized.xlsx")
pairs = pd.read_excel(DATA / "within_all_pairs_categorized.xlsx")

emit("\n--- Arm sizes (pairs file; preregistered 3 x 600) ---")
emit(pairs.groupby(["design", "fixed_game"])["pair_id"].nunique().to_string())
emit(f"total participants: {pairs['pair_id'].nunique()}")

first = long[long["trial_order"] == 1].set_index("pair_id")
second = long[long["trial_order"] == 2].set_index("pair_id")
common = first.index.intersection(second.index)
first, second = first.loc[common], second.loc[common]

sw = pd.DataFrame({
    "design": first["design"],
    "fixed_game": first["fixed_game"],
    "choice_switch": (~np.isclose(first["allocation"], second["allocation"])).astype(int),
    "cat_switch": (first["category_num"] != second["category_num"]).astype(int),
})

emit("\n--- Switching rates (validate vs audited v1.1 numbers) ---")
expected = {("control_kwlt", "pooled"): (46.2, 22.7),
            ("market_control", "kw"): (73.1, 51.6),
            ("market_control", "lt"): (82.3, 55.0)}
for (design, fg), grp in sw.groupby(["design", "fixed_game"]):
    cs, ks = 100 * grp["choice_switch"].mean(), 100 * grp["cat_switch"].mean()
    e = expected[(design, fg)]
    ok = abs(cs - e[0]) < 0.15 and abs(ks - e[1]) < 0.15
    emit(f"{design:<16} {fg:<7} N={len(grp):>4}  choice={cs:.1f}%  category={ks:.1f}%   "
         f"[{'OK' if ok else 'MISMATCH'} vs paper {e[0]}/{e[1]}]")

emit("\n--- LPM: choice switch on category switch (paper: 0.44 (0.04) / 0.19 (0.04) / 0.14 (0.03)) ---")
for (design, fg), grp in sw.groupby(["design", "fixed_game"]):
    fit = smf.ols("choice_switch ~ cat_switch", grp).fit(cov_type="HC1")
    emit(f"{design:<16} {fg:<7} b={fit.params['cat_switch']:.3f} (SE {fit.bse['cat_switch']:.3f}), "
         f"p={fit.pvalues['cat_switch']:.2e}")

# ---- full verification of AA's within regression tables under strict FE ----

ctrl = long[long["design"] == "control_kwlt"].copy()
ctrl["KW"] = (ctrl["game"] == "kw").astype(int)
ctrl["LTFirst"] = (ctrl["first_condition"] == "lt").astype(int)
mkt = long[long["design"] == "market_control"].copy()
mkt["Market"] = (mkt["condition"] == "market").astype(int)
mkt["ControlFirst"] = (mkt["first_condition"] == "control").astype(int)
for df in (ctrl, mkt):
    df["HighSP"] = (df["sp_num"] > 1).astype(int)
    df["cat"] = pd.Categorical(df["category"],
        ["Moral", "Mutual Benefit / Cooperation", "Self-interest", "No clear justification"])
    df["Second"] = (df["trial_order"] == 2).astype(int)


def fe(formula, df, keys, label, published):
    fit = smf.ols(formula + " + C(pair_id)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"]})
    out = "  ".join(f"{k.split('T.')[-1][:14]}={fit.params[k]:+.3f} ({fit.bse[k]:.3f})"
                    for k in keys if k in fit.params)
    emit(f"{label}: {out}  R2={fit.rsquared:.3f}")
    emit(f"   published: {published}")


emit("\n--- AA's within tables, strict FE (LSDV), clustered by participant ---")
fe("share_sent ~ KW + KW:LTFirst", ctrl, ["KW", "KW:LTFirst"],
   "ctrl col(1)", "KW=-0.036 (0.011), KWxLTFirst=-0.026 (0.016), R2=0.867")
fe("share_sent ~ KW + KW:LTFirst + HighSP", ctrl, ["KW", "KW:LTFirst", "HighSP"],
   "ctrl col(2)", "KW=-0.035 (0.011), KWxLTFirst=-0.026 (0.016), HighSP=+0.011 (0.015), R2=0.868")
fe("share_sent ~ KW + KW:LTFirst + C(cat)", ctrl,
   ["KW", "KW:LTFirst", "C(cat)[T.Mutual Benefit / Cooperation]", "C(cat)[T.Self-interest]",
    "C(cat)[T.No clear justification]"],
   "ctrl col(3)", "KW=-0.033 (0.010), KWxLTFirst=-0.033 (0.015), MBC=-0.070 (0.072), "
   "Self=-0.091 (0.026), NoClear=-0.053 (0.039), R2=0.877")
fe("share_sent ~ KW + KW:LTFirst + HighSP + C(cat)", ctrl,
   ["KW", "KW:LTFirst", "HighSP", "C(cat)[T.Mutual Benefit / Cooperation]", "C(cat)[T.Self-interest]",
    "C(cat)[T.No clear justification]"],
   "ctrl col(4)", "KW=-0.033 (0.010), KWxLTFirst=-0.033 (0.015), HighSP=-0.004 (0.015), "
   "MBC=-0.070 (0.072), Self=-0.093 (0.026), NoClear=-0.054 (0.040), R2=0.877")
fe("share_sent ~ Market + Market:ControlFirst", mkt, ["Market", "Market:ControlFirst"],
   "mkt col(1)", "Market=-0.160 (0.017), MktxCF=+0.016 (0.025), R2=0.597")
fe("share_sent ~ Market + Market:ControlFirst + HighSP", mkt, ["Market", "Market:ControlFirst", "HighSP"],
   "mkt col(2)", "Market=-0.139 (0.018), MktxCF=+0.017 (0.025), HighSP=+0.113 (0.021), R2=0.614")
fe("share_sent ~ Market + Market:ControlFirst + C(cat)", mkt,
   ["Market", "Market:ControlFirst", "C(cat)[T.Mutual Benefit / Cooperation]", "C(cat)[T.Self-interest]",
    "C(cat)[T.No clear justification]"],
   "mkt col(3)", "Market=-0.135 (0.018), MktxCF=+0.017 (0.024), MBC=-0.076 (0.033), "
   "Self=-0.197 (0.022), NoClear=-0.084 (0.034), R2=0.642")

fe("share_sent ~ Market + Market:ControlFirst + HighSP + C(cat)", mkt,
   ["Market", "Market:ControlFirst", "HighSP", "C(cat)[T.Mutual Benefit / Cooperation]",
    "C(cat)[T.Self-interest]", "C(cat)[T.No clear justification]"],
   "mkt col(4)", "Market=-0.126 (0.019), MktxCF=+0.017 (0.024), HighSP=+0.062 (0.021), "
   "MBC=-0.073 (0.033), Self=-0.177 (0.023), NoClear=-0.068 (0.034), R2=0.647")

emit("\nNOTE: the published LTFirst=+0.104 (0.008) and ControlFirst=+0.184 (0.013) rows were")
emit("time-invariant regressors, absorbed by participant FE. AA confirmed (2026-07-10) they")
emit("came from statsmodels' generalized (pseudo-inverse) solution to the rank-deficient")
emit("design: not identified, not to be interpreted. Rows dropped from the paper tables;")
emit("order effects are the no-FE participant-clustered estimates below (in the table notes).")

emit("\n--- Identified order effects (for the Appendix B prose) ---")
emit("cell means of share sent by arm x order group x decision:")
cells = long.groupby(["design", "first_condition", "trial_order"])["share_sent"].agg(["mean", "count"])
emit((100 * cells["mean"]).round(1).unstack("trial_order").to_string())
for name, df, treat, ordvar in (("ctrl", ctrl, "KW", "LTFirst"), ("mkt", mkt, "Market", "ControlFirst")):
    fit = smf.ols(f"share_sent ~ {treat} + Second + C(pair_id)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"]})
    emit(f"{name}: FE share ~ {treat} + Second:  Second={fit.params['Second']:+.3f} ({fit.bse['Second']:.3f})")
    lvl = smf.ols(f"share_sent ~ {treat} * {ordvar}", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"]})
    emit(f"{name}: pooled OLS level effect of {ordvar}: {lvl.params[ordvar]:+.3f} ({lvl.bse[ordvar]:.3f})")

# ----------------------------------------------------------------------------

TABLE_DIR.mkdir(exist_ok=True)
(TABLE_DIR / "llm_human_agreement.tex").write_text(table)
STATS_FILE.write_text("\n".join(LINES) + "\n")
emit(f"\nWrote {TABLE_DIR / 'llm_human_agreement.tex'} and {STATS_FILE}")
