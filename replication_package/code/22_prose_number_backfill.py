#!/usr/bin/env python3
"""Back every prose number that had no emitted log with a computed, logged value
(2026-07-20 Numbers-Reviewer sweep follow-up). Four items:

A. Section 3.1 footnote sample counts (P1 1,604/1,603/2,400/2,400 by game;
   P2 2,402/2,346; total 12,755; headline N~12,750).
B. Section 4.2: share of TG senders believing at least half of the tripled
   amount comes back (>= 1/2, per the censored-anchor convention):
   Control 30.5%, Market 52.0% (all senders with a hypothetical belief);
   classified-only Market 52.4%. Script 17 emits only the Control figure.
C. Section 4.2 fold-in footnote: leave-one-out accuracy and kappa on the 95
   reference examples at FULL precision (prose: 84%, kappa=0.72), from the
   stored per-example LOO predictions.
D. Section 4.2 fold-in footnote: exact max |folded - baseline| deviation of
   the Market-Control category-share effects (prose: "within 5 percentage
   points"; foldin_summary.txt displays 5.0).

Inputs:  data/player1_all_categorized.xlsx, data/player2_all_categorized.xlsx,
         output/unclassified/{loo_validation_claude-opus-4-8.csv,
         loo_validation_claude-sonnet-4-6.csv, player1_reference_examples.csv,
         classified_claude-opus-4-8.xlsx}
Outputs: output/tables/prose_number_backfill.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
UNCL = HERE.parent / "output" / "unclassified"
TABLES = HERE.parent / "output" / "tables"

CATS = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]

L: list[str] = []
n_fail = 0


def log(*s: object) -> None:
    L.append(" ".join(str(x) for x in s))
    print(L[-1])


def ok(name: str, cond: bool) -> None:
    global n_fail
    log(("  PASS  " if cond else "  FAIL  ") + name)
    n_fail += 0 if cond else 1


def kappa(true: np.ndarray, pred: np.ndarray) -> float:
    labs = sorted(set(true) | set(pred))
    po = float(np.mean(true == pred))
    pe = sum((np.mean(true == l)) * (np.mean(pred == l)) for l in labs)
    return (po - pe) / (1 - pe)


def main() -> None:
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1["story"] = pd.to_numeric(p1["story"], errors="coerce")
    p2 = pd.read_excel(DATA / "player2_all_categorized.xlsx")

    log("=== A. Sample counts (Section 3.1 footnote) ===")
    c1 = p1.groupby("game").size()
    c2 = p2.groupby("game").size()
    total = int(c1.sum() + c2.sum())
    for g, want in [("dgkw", 1604), ("dglt", 1603), ("ug", 2400), ("tg", 2400)]:
        ok(f"P1 {g}: {int(c1[g])} (footnote says {want})", int(c1[g]) == want)
    for g, want in [("ug", 2402), ("tg", 2346)]:
        ok(f"P2 {g}: {int(c2[g])} (footnote says {want})", int(c2[g]) == want)
    ok(f"total participants {total} (headline N~12,750)", total == 12755)

    log("\n=== B. TG senders with believed return >= 1/2 (Section 4.2) ===")
    tg = p1[(p1.game == "tg") & p1.beliefs_hp.notna()].copy()
    s = tg.beliefs_hp / 6.0 if tg.beliefs_hp.max() > 1.5 else tg.beliefs_hp
    tg["believes_half"] = s >= 0.5
    for story, lab, want in [(0, "Control", 30.5), (1, "Market", 52.0)]:
        got = 100 * tg.loc[tg.story == story, "believes_half"].mean()
        log(f"  {lab}: {got:.4f}% of TG senders with a belief (N={int((tg.story == story).sum())})")
        ok(f"prose {want}% ({lab})", round(got, 1) == want)
    got_cl = 100 * tg.loc[(tg.story == 1) & tg.category.isin(CATS), "believes_half"].mean()
    log(f"  Market, classified-only: {got_cl:.4f}%")
    ok("decision-log 52.4% (Market, classified-only)", round(got_cl, 1) == 52.4)

    log("\n=== C. Exact LOO accuracy and kappa on the 95 reference examples ===")
    ref = pd.read_csv(UNCL / "player1_reference_examples.csv", encoding="utf-8-sig")
    for model, acc_want in [("claude-opus-4-8", 0.842), ("claude-sonnet-4-6", 0.832)]:
        loo = pd.read_csv(UNCL / f"loo_validation_{model}.csv")
        m = ref.merge(loo, on="source_row", validate="1:1")
        assert len(m) == 95
        acc = float((m.training_code == m.loo_code).mean())
        k = kappa(m.training_code.to_numpy(), m.loo_code.to_numpy())
        log(f"  {model}: accuracy {acc:.6f}, kappa {k:.6f} (N=95)")
        ok(f"{model} accuracy rounds to {acc_want}", round(acc, 3) == acc_want)
        if model == "claude-opus-4-8":
            log(f"  -> paper prints kappa=0.72; exact {k:.6f} "
                f"({'>= 0.715: round-half-up to 0.72 is exact-value-backed' if k >= 0.715 else '< 0.715: paper should print 0.71'})")

    log("\n=== D. Exact fold-in deviations (Market-Control category-share effects, pp) ===")
    cls = pd.read_excel(UNCL / "classified_claude-opus-4-8.xlsx")
    cls["story"] = pd.to_numeric(cls["story"], errors="coerce")
    hard = cls[["PROLIFIC_PID", "game", "story", "examples_reclassified_category"]]
    refx = pd.read_csv(UNCL / "player1_reference_examples.csv", encoding="utf-8-sig")
    refx = (refx[["PROLIFIC_PID", "game", "story", "training_category"]]
            .rename(columns={"training_category": "examples_reclassified_category"}))
    forced = pd.concat([hard, refx], ignore_index=True)
    assert len(forced) == 309, f"expected 309 forced rows (214 hard + 95 reference), got {len(forced)}"
    base = p1[p1.story.isin([0, 1])].copy()
    m = base.merge(forced, on=["PROLIFIC_PID", "game", "story"], how="left")
    m["cat_folded"] = m.category.where(m.category.isin(CATS), m.examples_reclassified_category)

    def effects(d: pd.DataFrame, col: str) -> pd.DataFrame:
        d = d[d[col].isin(CATS)]
        sh = (d.groupby(["game", "story"])[col].value_counts(normalize=True)
              .rename("q").reset_index())
        piv = sh.pivot_table(index=["game", col], columns="story", values="q", fill_value=0.0)
        return 100 * (piv[1] - piv[0])

    eff_base = effects(m, "category")
    eff_fold = effects(m, "cat_folded")
    dev = (eff_fold - eff_base).abs()
    worst = dev.idxmax()
    log(f"  max |folded - baseline| = {dev.max():.4f} pp at {worst}")
    for cell, v in dev.sort_values(ascending=False).head(3).items():
        log(f"    {cell}: baseline {eff_base[cell]:+.2f}, folded {eff_fold[cell]:+.2f}, |dev| {v:.4f}")
    ok('prose "within 5 percentage points" (exact max < 5.0)', dev.max() < 5.0)
    ok("worst cell is (dgkw, Self-interest)", worst == ("dgkw", "Self-interest"))

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "prose_number_backfill.txt").write_text("\n".join(L) + "\n")
    print(f"\n{'ALL CHECKS PASS' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}"
          f" — wrote {TABLES / 'prose_number_backfill.txt'}")


if __name__ == "__main__":
    main()
