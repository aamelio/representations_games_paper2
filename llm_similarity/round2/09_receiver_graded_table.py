#!/usr/bin/env python3
"""Receiver-vignette graded-similarity table (NG meeting 2026-07-23, item 4).

Reads recording_receiver_graded.csv (from 08_receiver_graded_runner.py), maps vignette
ids to receiver types (UA=Accepting, UR=Rejecting, TR=Returning, TK=Keeping), averages
the graded ratings over the type's two vignettes x three conversations (6 ratings per
cell), and writes the paper table. Also compares the re-elicited splits with the
original in-conversation splits (recording_splits_round2.csv) as a test-retest check.

Headline rater only (claude = Opus 4.8), per the round-2 protocol.

Output: ../../replication_package/output/tables/similarity_receivers.tex
        out/receiver_graded_stats.txt
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent.parent / "replication_package" / "output" / "tables"
OUT = HERE / "out"

TYPE = {"UA": "Accepting", "UR": "Rejecting", "TR": "Returning", "TK": "Keeping"}
CONTEXT_ROWS = {"C-UG": "UG, Control", "M-UG": "UG, Market",
                "C-TG": "TG, Control", "M-TG": "TG, Market"}
GAME_TYPES = {"UG": ["Accepting", "Rejecting"], "TG": ["Returning", "Keeping"]}
HEADLINE = "claude"


def load() -> pd.DataFrame:
    f = HERE / "recording_receiver_graded.csv"
    assert f.exists(), "recording_receiver_graded.csv missing - run 08_receiver_graded_runner.py"
    df = pd.read_csv(f)
    df = df[df.model == HEADLINE].copy()
    df["rtype"] = df["item_id"].str[:2].map(TYPE)
    n_sets = df["set"].nunique()
    assert n_sets == 3, f"expected 3 conversations for the headline rater, found {n_sets}"
    return df


def main() -> None:
    df = load()
    g = df[df.task == "graded"]
    cells = g.groupby(["context", "rtype"])["value"].agg(["mean", "count"])
    assert (cells["count"] == 6).all(), "expected 2 vignettes x 3 sets = 6 ratings per cell"
    mean = cells["mean"].unstack()

    lines = ["Receiver-vignette graded similarity, headline rater, sets 1-3", "",
             mean.round(1).to_string(), "",
             "Market - Control deltas (graded):"]
    for game, types in GAME_TYPES.items():
        for t in types:
            delta = mean.loc[f"M-{game}", t] - mean.loc[f"C-{game}", t]
            lines.append(f"  {game} {t}: {delta:+.1f}")

    sp_new = (df[df.task == "split"].groupby(["context", "rtype"])["value"].sum()
              / (3 * 100) * 100).unstack()
    old = pd.read_csv(HERE / "recording_splits_round2.csv")
    old = old[(old.model == HEADLINE) & (old.task == "receiver")].copy()
    old["rtype"] = old["item_id"].str[:2].map(TYPE)
    sp_old = (old.groupby(["context", "rtype"])["points"].sum() / (3 * 100) * 100).unstack()
    lines += ["", "Split shares, re-elicited (this module) vs original (in-conversation):"]
    for ctx in CONTEXT_ROWS:
        game = "UG" if "UG" in ctx else "TG"
        t0 = GAME_TYPES[game][0]
        lines.append(f"  {ctx} {t0}: {sp_new.loc[ctx, t0]:.1f} vs {sp_old.loc[ctx, t0]:.1f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "receiver_graded_stats.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    def cell(ctx: str, t: str) -> str:
        return f"{mean.loc[ctx, t]:.1f}"

    rows = []
    for game in ["UG", "TG"]:
        t1, t2 = GAME_TYPES[game]
        rows.append(rf"\multicolumn{{3}}{{l}}{{\emph{{{'Ultimatum game' if game == 'UG' else 'Trust game'}: {t1} vs.\ {t2}}}}} \\")
        for cond, key in [("Control", f"C-{game}"), ("Market", f"M-{game}")]:
            rows.append(f"{cond} & {cell(key, t1)} & {cell(key, t2)} \\\\")
        d1 = mean.loc[f"M-{game}", t1] - mean.loc[f"C-{game}", t1]
        d2 = mean.loc[f"M-{game}", t2] - mean.loc[f"C-{game}", t2]
        rows.append(rf"Market $-$ Control & {d1:+.1f} & {d2:+.1f} \\")
        if game == "UG":
            rows.append(r"\midrule")
    tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Similarity of Strategic-Game Contexts to Receiver-Type Vignettes}}
\label{tab:similarity_receivers}
\begin{tabular}{l cc}
\toprule
Context & \multicolumn{2}{c}{Graded similarity (0--100)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: Mean 0--100 ratings of how similar the counterpart evoked by each context is to the receiver-type vignettes (two vignettes per type, three permuted-label conversations; six ratings per cell), from the round-2 headline rater under the frozen protocol of Appendix~\ref{app:similarity_vignettes}. Contexts are the verbatim Control and Market instructions of the ultimatum and trust games. The forced 100-point splits over the same vignettes are in Table~\ref{tab:receiver_splits}.
\end{flushleft}
\end{table}
"""
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "similarity_receivers.tex").write_text(tex)
    print(f"\nwrote {TABLES / 'similarity_receivers.tex'}")


if __name__ == "__main__":
    main()
