#!/usr/bin/env python3
"""Formal checks on AA's 48-vignette sender pool (task 5 recap numbers).

Regenerates the checks quoted in Section 5 of post_meeting_tasks.tex: vignette count,
banned words in vignette bodies, distinct protagonist names, individuals/firms balance
within every category x belief cell, and word-count range.

Input:   ../vignettes_6rep_draft.md
Output:  sender_pool_checks.txt
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent / "vignettes_6rep_draft.md"

BANNED = re.compile(
    r"\b(fair|unfair|selfish|generous|greedy|cooperat\w*|moral\w*|trust\w*|betray\w*)\b",
    re.IGNORECASE)

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def main() -> None:
    text = DRAFT.read_text()
    items = re.findall(r"^\*\*([A-Z]{2}\d)\s*[—–-]+\s*([^.]+)\.\*\*\s*(.+)$", text, re.M)
    log(f"vignettes found: {len(items)} (expected 48)")

    banned_hits, names, counts = [], [], {}
    for vid, _setting, body in items:
        hits = BANNED.findall(body)
        if hits:
            banned_hits.append((vid, hits))
        names.append(body.split()[0].rstrip(",'s"))
        counts[vid] = len(body.split())

    log(f"banned words in bodies: {banned_hits if banned_hits else 'none'}")
    log(f"distinct protagonist names: {len(set(names))} of {len(names)}")

    cells = sorted({vid[:2] for vid, _, _ in items})
    log(f"cells: {cells} (expected 6: CH CL MH ML SH SL)")
    for c in cells:
        ids = sorted(int(vid[2]) for vid, _, _ in items if vid.startswith(c))
        individuals = [i for i in ids if i <= 4]
        firms = [i for i in ids if i >= 5]
        log(f"  {c}: {len(ids)} vignettes, individuals {len(individuals)}, firms {len(firms)}")

    log(f"word counts: min {min(counts.values())}, max {max(counts.values())}")

    (HERE / "sender_pool_checks.txt").write_text("\n".join(LOG))
    print(f"\nwrote {HERE / 'sender_pool_checks.txt'}")


if __name__ == "__main__":
    main()
