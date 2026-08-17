#!/usr/bin/env python3
"""Dump paste-ready prompts for the manual GPT/Gemini arms of the neutral-prompt rerun.

Writes manual_prompts/: one Prompt-1 file per triplet (game and story texts inserted in
the triplet's label order, from rerun_materials.json) plus prompt2.txt and prompt3.txt,
identical to what the API Claude arm received (rerun_runner.py), so the three arms of
the rerun differ only in the rating model. See manual_prompts/README.md for the
session-by-session protocol.
"""

from pathlib import Path

import rerun_runner as rr

HERE = Path(__file__).resolve().parent
OUT = HERE / "manual_prompts"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mat = rr.load_materials()
    for t in rr.TRIPLETS:
        (OUT / f"triplet{t}_prompt1.txt").write_text(rr.prompt1(t, mat))
    (OUT / "prompt2.txt").write_text(rr.prompt2())
    (OUT / "prompt3.txt").write_text(rr.prompt3())
    for f in sorted(OUT.glob("*.txt")):
        print(f"wrote {f.relative_to(HERE)} ({len(f.read_text().split())} words)")


if __name__ == "__main__":
    main()
