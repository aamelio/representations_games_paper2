#!/usr/bin/env python3
"""Task 7: regenerate the pilot and second-rater statistics quoted in the recap note.

The pilot means/SDs and the Opus-vs-Sonnet agreement statistics of Section 7
(post_meeting_tasks.tex) were originally printed at run time; this script recomputes
them from the saved score files so every quoted number has a generating script.

Inputs:  pilot_scores.csv, agreement_scores_claude-sonnet-5.csv, summaries_p1.csv
Output:  out/pilot_agreement_stats.txt
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ASPECTS = ["F", "O", "J"]

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def main() -> None:
    pilot = pd.read_csv(HERE / "pilot_scores.csv")
    alt = pd.read_csv(HERE / "agreement_scores_claude-sonnet-5.csv")

    log(f"=== pilot (Opus 4.8, N={len(pilot)}) ===")
    log("pooled means by story:")
    log(pilot.groupby("story")[ASPECTS].mean().round(1).to_string())
    sds = pilot.groupby(["game", "story"])[ASPECTS].std()
    log(f"\nwithin-cell SD range: {sds.min().min():.1f} to {sds.max().max():.1f}")

    m = pilot.merge(alt, on="PROLIFIC_PID", suffixes=("_opus", "_alt"))
    log(f"\n=== second-rater check (Sonnet 5 vs Opus 4.8, N={len(m)}) ===")
    for a in ASPECTS:
        x, y = m[f"{a}_opus"], m[f"{a}_alt"]
        log(f"  {a}: pearson r={x.corr(y):.3f}  mean abs diff={(x - y).abs().mean():.1f}  "
            f"mean opus={x.mean():.1f} sonnet={y.mean():.1f}")
    log(f"  dominant-aspect agreement: {(m['dominant_opus'] == m['dominant_alt']).mean():.0%}")
    log("\nAid-Bonus contrast under the second rater (direction check):")
    log(m.groupby("story")[[f"{a}_alt" for a in ASPECTS]].mean().round(1).to_string())

    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "pilot_agreement_stats.txt").write_text("\n".join(LOG))
    print(f"\nwrote {HERE / 'out' / 'pilot_agreement_stats.txt'}")


if __name__ == "__main__":
    main()
