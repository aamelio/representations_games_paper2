#!/usr/bin/env python3
"""Generate tab:llm_similarity (round-1 story-to-game similarity, Section 3.3).

The table was previously hand-typed in main.tex from the round-1 recording
workbook; this script closes the reproducibility gap: it recomputes every cell
from the raw rating sheets (P1_Similarity: 72 story-game ratings; P3_Retrieval:
36 100-point splits; 9 conversations = 3 models x 3 permuted-label triplets),
asserts the means against the workbook's own Summary sheet, and emits the
table verbatim as previously typed (similarity to 0.1, retrieval shares to
integers).

Inputs:  data/memory_games_llm_recording.xlsx  (copy of
         llm_similarity/memory_games_llm_recording.xlsx, the audited original)
Outputs: output/tables/llm_similarity.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TABLES = HERE.parent / "output" / "tables"

GAMES = ["DG-KW", "DG-LT", "UG", "TG"]


def main() -> None:
    xl = pd.ExcelFile(DATA / "memory_games_llm_recording.xlsx")
    sim = pd.read_excel(xl, "P1_Similarity")
    ret = pd.read_excel(xl, "P3_Retrieval")
    summary = pd.read_excel(xl, "Summary", header=None)

    assert len(sim) == 72 and len(ret) == 36, "unexpected raw sheet sizes"
    assert sim.groupby(["Game", "Story"]).size().eq(9).all(), "expected 9 ratings per cell"

    s = sim.groupby(["Game", "Story"])["Similarity (0–100)"].mean().unstack()
    s["Mean"] = s[["Bonus", "Aid"]].mean(axis=1)
    r = ret.groupby("Game")[["Bonus share (0–100)", "Aid share (0–100)"]].mean()

    # Cross-check the recomputed means against the workbook's Summary tables
    # (Table 1 rows 4-7: Bonus/Aid/Row mean; Table 3 rows 20-23: Bonus/Aid share).
    for i, g in enumerate(GAMES):
        want = summary.iloc[4 + i, 1:4].astype(float).to_numpy()
        got = s.loc[g, ["Bonus", "Aid", "Mean"]].to_numpy()
        assert np.allclose(got, want, atol=1e-6), f"Summary mismatch (similarity, {g})"
        want_r = summary.iloc[20 + i, 1:3].astype(float).to_numpy()
        got_r = r.loc[g].to_numpy()
        assert np.allclose(got_r, want_r, atol=1e-6), f"Summary mismatch (retrieval, {g})"
        print(f"  [validate] {g}: similarity {got.round(1)} retrieval {got_r.round(1)}  OK")

    body = []
    for g in GAMES:
        body.append(
            f"{g:<5s} & {s.loc[g, 'Bonus']:.1f} & {s.loc[g, 'Aid']:.1f} & "
            f"{s.loc[g, 'Mean']:.1f} & {round(r.loc[g, 'Bonus share (0–100)'])} / "
            f"{round(r.loc[g, 'Aid share (0–100)'])} \\\\")
    tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Perceived Similarity of Stories to Games, LLM Raters}}
\label{tab:llm_similarity}
\begin{tabular}{l ccc c}
\toprule
& \multicolumn{3}{c}{Structural similarity (0--100)} & Retrieval split \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
Game & Bonus & Aid & Mean & Bonus / Aid \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Means over nine conversations (three models $\times$ three label permutations). The games were presented under neutral labels (Games A--D), their texts the verbatim Control-condition instructions; the stories as Story 1 and Story 2. Structural similarity: 0--100 rating of each story--game pair, under an instruction to judge the strategic structure of the two situations and ignore surface thematic content. Retrieval split: 100 points divided between the two stories according to which the game calls to mind as a similar past experience, with no restriction on what may drive recall.
\end{flushleft}
\end{table}
"""
    (TABLES / "llm_similarity.tex").write_text(tex)
    print(f"wrote {TABLES / 'llm_similarity.tex'}")


if __name__ == "__main__":
    main()
