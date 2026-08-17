#!/usr/bin/env python3
"""Receiver-type splits table (tab:receiver_splits), added 2026-07-22 per SN.

Aggregates the third rating task of the round-2 protocol --- for each strategic
context, 100 points distributed across the four counterpart vignettes (two per
receiver type) --- by type, over the three permuted conversations of the headline
rater. The accepting/returning columns are the numbers quoted in Section 3.3
prose (42.7 -> 52.0; 53.0 -> 54.0); computed from the raw recording, NOT from
out/receiver_splits.csv, whose per-item means are pre-rounded (double-rounding
hazard documented 2026-07-20).

Input:  recording_splits_round2.csv  (task == "receiver", model == "claude")
Output: ../../replication_package/output/tables/receiver_splits.tex
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent / "replication_package"
OUT = PKG / "output" / "tables" / "receiver_splits.tex"

splits = pd.read_csv(HERE / "recording_splits_round2.csv")
rec = splits[(splits.model == "claude") & (splits.task == "receiver")].copy()


def share(ctx: str, prefix: str) -> float:
    d = rec[rec.context == ctx]
    return 100.0 * d.loc[d.item_id.str.startswith(prefix), "points"].sum() / d.points.sum()


rows = {
    "Control": (share("C-UG", "UA"), share("C-UG", "UR"),
                share("C-TG", "TR"), share("C-TG", "TK")),
    "Market": (share("M-UG", "UA"), share("M-UG", "UR"),
               share("M-TG", "TR"), share("M-TG", "TK")),
}

# each game's pair must sum to 100; displayed values must match the Section 3.3 prose
for cond, (ua, ur, tr, tk) in rows.items():
    assert abs(ua + ur - 100.0) < 1e-9 and abs(tr + tk - 100.0) < 1e-9, cond
assert f"{rows['Control'][0]:.1f}" == "42.7" and f"{rows['Market'][0]:.1f}" == "52.0"
assert f"{rows['Control'][2]:.1f}" == "53.0" and f"{rows['Market'][2]:.1f}" == "54.0"

body = "\n".join(
    f"{cond} & {ua:.1f} & {ur:.1f} & {tr:.1f} & {tk:.1f} \\\\"
    for cond, (ua, ur, tr, tk) in rows.items()
)

tex = r"""\begin{table}[!htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.15}
\caption{\textbf{Similarity of Strategic Contexts to Receiver Types, LLM Raters}}
\label{tab:receiver_splits}
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{Ultimatum game} & \multicolumn{2}{c}{Trust game} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Condition & Accepting & Rejecting & Returning & Keeping \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Notes: For each strategic context, raters distributed 100 points across the game's four counterpart vignettes---two per receiver type: counterparts who accept or reject one-sided terms (ultimatum game) and counterparts who return or keep entrusted proceeds (trust game)---according to which counterpart they would picture on the other side. Cells aggregate the points by type and average over the three permuted conversations of the headline rater (protocol of Table~\ref{tab:similarity_categories}); each game's pair sums to 100. The accepting and returning columns are the splits quoted in Section~\ref{sec:similarity}.
\end{flushleft}
\end{table}
"""
OUT.write_text(tex)
print(f"wrote {OUT}")
for cond, (ua, ur, tr, tk) in rows.items():
    print(f"  {cond:>8}: UG {ua:.1f}/{ur:.1f}  TG {tr:.1f}/{tk:.1f}")
