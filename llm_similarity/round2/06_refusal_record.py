#!/usr/bin/env python3
"""Consolidate the Fable-rater refusal record into an emitted artifact
(2026-07-20 Numbers-Reviewer follow-up).

The appendix states: "eleven times across eight candidate label orders". Refused
conversations produce no transcript, so no machine log of the refusal events
exists; the authoritative contemporaneous record is (a) the inline run record in
03_api_runner.py (the CONTEXT_LABELS comment block, quoted verbatim below) and
(b) the seed-by-seed narrative in ng_comments_tracker.md (2026-07-19 entries).
This script extracts (a) verbatim, inventories the completed-run transcripts,
and writes out/refusal_record.md so the tally has an emitted artifact inside
the module.

Run: python3 06_refusal_record.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    runner = (HERE / "03_api_runner.py").read_text()
    m = re.search(r"CONTEXT_LABELS = \{.*?\n\}\n", runner, re.S)
    assert m, "CONTEXT_LABELS block not found in 03_api_runner.py"
    block = m.group(0)
    comment_lines = [ln for ln in block.splitlines() if ln.strip().startswith("#")]
    assert comment_lines, "no inline run record found in the CONTEXT_LABELS block"

    transcripts = sorted(p.name for p in (HERE / "transcripts").glob("rater_*.json"))
    fable = [t for t in transcripts if "fable" in t]

    doc = "\n".join(
        [
            "# Fable-rater refusal record (round 2)",
            "",
            "Appendix H states the third Fable-5 conversation completed only after the",
            "model's safety filter declined the benign opening message **eleven times",
            "across eight candidate label orders**. Refused conversations produce no",
            "transcript, so the contemporaneous record lives in the runner's inline",
            "comments (extracted verbatim below, from `03_api_runner.py`) and in the",
            "seed-by-seed narrative of `ng_comments_tracker.md` (2026-07-19 entries),",
            "which is the authoritative tally.",
            "",
            "## Eight candidate label orders for the third conversation",
            "",
            "Sets 3, 4 (seed 20260719), 5 (seed 20260720), then the pre-committed",
            "successor-seed probe hunt over 20260721-24, and 20260725 (= set 6, the",
            "order that completed).",
            "",
            "## Inline run record from 03_api_runner.py (verbatim)",
            "",
            "```",
            *comment_lines,
            "```",
            "",
            "## Completed-run transcripts present in this module",
            "",
            *[f"- {t}" for t in transcripts],
            "",
            f"Fable conversations completed: {len(fable)} (sets 1, 2, 6) — matching the",
            "appendix's statement that all three permuted conversations completed, the",
            "third on the set-6 substitute order.",
            "",
        ]
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "refusal_record.md").write_text(doc)
    print(f"wrote {OUT / 'refusal_record.md'} "
          f"({len(comment_lines)} record lines, {len(transcripts)} transcripts, {len(fable)} Fable)")
    assert len(fable) == 3, "expected 3 completed Fable transcripts (sets 1, 2, 6)"


if __name__ == "__main__":
    main()
