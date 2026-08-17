"""Fold the forced classifications of the residual Player 1 answers back into the
Market-vs-Control category shares; produces every number cited in the robustness
footnote of Section 4.2 (sec:market_control).

Inputs: data/player1_all_categorized.xlsx, data/player1_control_market_no_clear_reclassified.xlsx,
and the outputs of 18_unclassified_classification.py (reference examples, classified_<model>.xlsx,
loo_validation_<model>.csv). Headline model claude-opus-4-8; claude-sonnet-4-6 and the earlier
forced LLM pass (reclassification_code) enter as sensitivity variants.

Output: output/unclassified/foldin_summary.txt (+ printed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "output" / "unclassified"

HEADLINE = "claude-opus-4-8"
CROSS = "claude-sonnet-4-6"
CATS = {1: "Moral", 2: "Mutual Benefit / Cooperation", 3: "Self-interest"}
GAMES = ["dgkw", "dglt", "ug", "tg"]
LOG: list[str] = []


def log(*a: object) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def forced_labels(llm_wb: pd.DataFrame, ref: pd.DataFrame, model: str) -> pd.Series:
    """Forced code per source_row: training label for the 95, model prediction otherwise."""
    pred = pd.read_excel(OUT / f"classified_{model}.xlsx")
    pred["examples_reclassification_code"] = pd.to_numeric(
        pred["examples_reclassification_code"], errors="coerce")
    out = llm_wb[["source_row"]].merge(
        ref[["source_row", "training_code"]], on="source_row", how="left").merge(
        pred[["source_row", "examples_reclassification_code"]], on="source_row", how="left")
    return out.training_code.fillna(out.examples_reclassification_code)


def market_effects(db: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = {}
    for g in GAMES:
        sub = db[db.game.eq(g)].dropna(subset=[col])
        tab = pd.crosstab(sub[col], sub.story, normalize="columns") * 100
        rows[g] = tab[1] - tab[0]
    return pd.DataFrame(rows).T[list(CATS.values())]


def loo_stats(model: str, ref: pd.DataFrame) -> str:
    db = pd.read_csv(OUT / f"loo_validation_{model}.csv").merge(
        ref[["source_row", "training_code"]], on="source_row", validate="one_to_one")
    db["loo_code"] = pd.to_numeric(db.loo_code, errors="coerce")
    acc = db.loo_code.eq(db.training_code).mean()
    a, b = db.training_code, db.loo_code
    pe = sum((a == k).mean() * (b == k).mean() for k in CATS)
    return f"accuracy {acc:.3f}, kappa {(acc - pe) / (1 - pe):.3f} (N={len(db)})"


def main() -> None:
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1 = p1[p1.story.isin([0, 1])].copy()
    llm_wb = pd.read_excel(DATA / "player1_control_market_no_clear_reclassified.xlsx")
    ref = pd.read_csv(OUT / "player1_reference_examples.csv")

    # --- 0. residual shares -----------------------------------------------------------
    unc0 = p1.category_num.eq(0)
    n = p1.groupby("story").size()
    nu = p1[unc0].groupby("story").size()
    log(f"[0] Residual answers: Control {nu[0]}/{n[0]} = {100*nu[0]/n[0]:.1f}%, "
        f"Market {nu[1]}/{n[1]} = {100*nu[1]/n[1]:.1f}%")

    # --- 1. LOO validation ------------------------------------------------------------
    log(f"\n[1] Leave-one-out validation on the 95 labelled examples:")
    for model in [HEADLINE, CROSS]:
        log(f"  {model}: {loo_stats(model, ref)}")

    # --- 2. fold-in under three labelings ---------------------------------------------
    variants = {
        f"headline ({HEADLINE})": forced_labels(llm_wb, ref, HEADLINE),
        f"cross-model ({CROSS})": forced_labels(llm_wb, ref, CROSS),
        "earlier forced LLM pass": llm_wb.reclassification_code,
    }
    p1["cat_orig"] = p1.category_num.map(CATS)
    base = market_effects(p1, "cat_orig")
    log("\n[2] Market-Control effects on category shares (pp), baseline = paper "
        "(residual dropped):")
    log(base.round(1).to_string())
    for name, codes in variants.items():
        lab = llm_wb[["PROLIFIC_PID", "game", "story"]].copy()
        lab["forced_cat"] = pd.Series(codes).map(CATS).to_numpy()
        m = p1.merge(lab, on=["PROLIFIC_PID", "game", "story"], how="left",
                     validate="one_to_one")
        m["cat_folded"] = m.cat_orig.where(~unc0.to_numpy(), m.forced_cat)
        n_lab = m.loc[unc0.to_numpy(), "forced_cat"].notna().sum()
        eff = market_effects(m, "cat_folded")
        dev = (eff - base).abs()
        log(f"\n  folded in, {name} ({n_lab}/309 residual labelled):")
        log("  " + eff.round(1).to_string().replace("\n", "\n  "))
        log(f"  max |deviation from baseline| = {dev.max().max():.1f} pp "
            f"(cell: {dev.stack().idxmax()})")

    # --- 3. composition of the residual under the headline labels ---------------------
    lab = llm_wb.copy()
    lab["forced_cat"] = forced_labels(llm_wb, ref, HEADLINE).map(CATS)
    log("\n[3] Forced-category composition of the residual answers, headline model (%):")
    log((pd.crosstab(lab.forced_cat, lab.story, normalize="columns") * 100)
        .round(1).to_string())

    # --- 4. the account: counterpart mentions, length, behavior -----------------------
    nomention = "No mention of recipient"
    log("\n[4] Share of answers mentioning no counterpart (%):")
    for story, name in [(0, "Control"), (1, "Market")]:
        r = p1[p1.story.eq(story) & unc0].social_proximity.eq(nomention).mean()
        c = p1[p1.story.eq(story) & ~unc0].social_proximity.eq(nomention).mean()
        log(f"  {name}: residual {100*r:.1f} vs classified {100*c:.1f}")
    p1["n_words"] = p1.reasons.astype(str).str.split().str.len()
    w = p1.groupby(["story", unc0.rename("residual")]).n_words.mean()
    log(f"\nMean words: Control classified {w[0][False]:.1f} / residual {w[0][True]:.1f}; "
        f"Market classified {w[1][False]:.1f} / residual {w[1][True]:.1f}")
    log("\nMean share sent (games with share_sent), Market, residual vs classified by game:")
    mk = p1[p1.story.eq(1)]
    log(mk.groupby(["game", unc0.rename("residual")]).share_sent.mean().round(3).to_string())

    # --- 5. cross-method agreement on the 214 hard cases ------------------------------
    hard = ~llm_wb.source_row.isin(ref.source_row)
    h, c, e = (forced_labels(llm_wb, ref, HEADLINE)[hard],
               forced_labels(llm_wb, ref, CROSS)[hard],
               llm_wb.reclassification_code[hard])
    log(f"\n[5] Agreement on the {hard.sum()} non-reference residual answers: "
        f"headline vs cross-model {h.eq(c).mean():.3f}; "
        f"headline vs earlier pass {h.eq(e).mean():.3f}; "
        f"cross-model vs earlier pass {c.eq(e).mean():.3f}")

    (OUT / "foldin_summary.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\nSaved {OUT / 'foldin_summary.txt'}")


if __name__ == "__main__":
    main()
