#!/usr/bin/env python3
"""Round-1 neutral-prompt rerun: comparison with the May round (task 3 recap numbers).

Regenerates every number in Section 3 of post_meeting_tasks.tex: per-game similarity
means (May all-model and per arm; rerun per arm and pooled), absolute Bonus-Aid gaps,
Bonus retrieval shares, paper Table-5 layout panels, and per-triplet ranges. The rerun
CSVs may contain one, two, or three arms; every rerun statistic is computed per model
(the arms ran at different dates/models), plus a pooled panel when more than one arm
is complete.

Inputs:  memory_games_llm_recording.xlsx (May), rerun_recording.csv, rerun_splits.csv
Output:  rerun_analysis_stats.txt
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GAMES = ["DG-KW", "DG-LT", "UG", "TG"]
MAY_CLAUDE = "Claude Opus 4.7"

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def sim_block(d: pd.DataFrame) -> None:
    for g in GAMES:
        log(f"    {g:6} Bonus {d.loc[g, 'Bonus']:.1f}  Aid {d.loc[g, 'Aid']:.1f}  "
            f"story-avg {d.loc[g, ['Bonus', 'Aid']].mean():.1f}  "
            f"|Bonus-Aid| {abs(d.loc[g, 'Bonus'] - d.loc[g, 'Aid']):.1f}")


def main() -> None:
    xl = pd.ExcelFile(HERE / "memory_games_llm_recording.xlsx")
    sim = pd.read_excel(xl, "P1_Similarity")
    ret = pd.read_excel(xl, "P3_Retrieval")
    rr = pd.read_csv(HERE / "rerun_recording.csv")
    rs = pd.read_csv(HERE / "rerun_splits.csv")
    rr_models = sorted(rr.model.unique())

    def may_sim(model: str | None) -> pd.DataFrame:
        d = sim if model is None else sim[sim.Model == model]
        return d.groupby(["Game", "Story"])["Similarity (0–100)"].mean().unstack().reindex(GAMES)

    def may_ret(model: str | None) -> pd.Series:
        d = ret if model is None else ret[ret.Model == model]
        return d.groupby("Game")["Bonus share (0–100)"].mean().reindex(GAMES)

    def rr_sim(model: str | None) -> pd.DataFrame:
        d = rr if model is None else rr[rr.model == model]
        return d.groupby(["game", "story"])["rating"].mean().unstack().reindex(GAMES)

    def rr_ret(model: str | None) -> pd.Series:
        d = rs if model is None else rs[rs.model == model]
        return d[d.story == "Bonus"].groupby("game")["points"].mean().reindex(GAMES)

    log("Similarity means by game x story (Bonus, Aid):")
    log("  May, all models:")
    sim_block(may_sim(None))
    for m in sorted(sim.Model.unique()):
        log(f"  May, {m}:")
        sim_block(may_sim(m))
    for m in rr_models:
        log(f"  Rerun (neutral prompt), {m}:")
        sim_block(rr_sim(m))
    if len(rr_models) > 1:
        log(f"  Rerun (neutral prompt), pooled over {len(rr_models)} arms:")
        sim_block(rr_sim(None))

    log("\nBonus retrieval share by game:")
    log("  May, all models:  " + "  ".join(f"{g} {may_ret(None)[g]:.0f}" for g in GAMES))
    for m in sorted(ret.Model.unique()):
        log(f"  May, {m}: " + "  ".join(f"{g} {may_ret(m)[g]:.0f}" for g in GAMES))
    for m in rr_models:
        log(f"  Rerun, {m}: " + "  ".join(f"{g} {rr_ret(m)[g]:.0f}" for g in GAMES))

    # Paper Table-5 layout: Bonus, Aid, Mean similarity + retrieval split, one panel per arm
    log("\nPaper Table-5 layout panels (similarity Bonus / Aid / Mean; retrieval split):")
    panels = [("May, guided prompt, three models x three permutations",
               may_sim(None), may_ret(None)),
              (f"May, guided prompt, {MAY_CLAUDE} arm", may_sim(MAY_CLAUDE), may_ret(MAY_CLAUDE)),
              ("May, guided prompt, ChatGPT 5.5 Thinking arm",
               may_sim("ChatGPT 5.5 Thinking"), may_ret("ChatGPT 5.5 Thinking")),
              ("May, guided prompt, Gemini 3 Pro arm",
               may_sim("Gemini 3 Pro"), may_ret("Gemini 3 Pro"))]
    panels += [(f"Rerun, neutral prompt, {m}", rr_sim(m), rr_ret(m)) for m in rr_models]
    if len(rr_models) > 1:
        panels.append((f"Rerun, neutral prompt, pooled over {len(rr_models)} arms",
                       rr_sim(None), rr_ret(None)))
    for lab, d, rb in panels:
        log(f"  {lab}:")
        for g in GAMES:
            b, a = d.loc[g, "Bonus"], d.loc[g, "Aid"]
            log(f"    {g:6} Bonus {b:5.1f}  Aid {a:5.1f}  Mean {(b + a) / 2:5.1f}  "
                f"split {rb[g]:.0f} / {100 - rb[g]:.0f}")

    log("\nPer-triplet ranges (max-min across the three conversations of an arm):")
    for m in rr_models:
        rng = (rr[rr.model == m].groupby(["game", "story"])["rating"]
               .agg(lambda s: s.max() - s.min()).unstack().reindex(GAMES))
        log(f"  Rerun, {m}:")
        log(rng.round(0).to_string())

    (HERE / "rerun_analysis_stats.txt").write_text("\n".join(LOG))
    print(f"\nwrote {HERE / 'rerun_analysis_stats.txt'}")


if __name__ == "__main__":
    main()
