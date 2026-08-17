#!/usr/bin/env python3
"""Graded context-to-receiver-vignette ratings (NG meeting 2026-07-23, item 4).

Round 2 elicited only 100-point splits over the receiver-type vignettes (Table 40);
NG asked for a graded similarity panel to sit beside the sender-category block of the
context-similarity table. This extension collects, for each of the four strategic
contexts (C-UG, M-UG, C-TG, M-TG), a 0-100 match rating for each of the four
game-relevant counterpart descriptions, plus a re-elicitation of the 100-point split
(test-retest against the original module, which ran inside the full conversations).

Protocol: headline rater only (Opus 4.8), sets 1-3, fresh conversation per set, one
message per strategic context in the set's TEXT-letter order, counterpart labels R1-R4
in the set's frozen permutation (03_api_runner.COUNTERPART_ORDER). Materials and
infrastructure imported from 03_api_runner.py; nothing in the frozen protocol changes.

Usage:
  python3 08_receiver_graded_runner.py --dry-run          # print set-1 prompts, no API calls
  python3 08_receiver_graded_runner.py                    # run (needs ANTHROPIC_API_KEY)
  python3 08_receiver_graded_runner.py --models claude_fable --sets 1  # robustness subset

Output: recording_receiver_graded.csv (model,set,task,context,item_id,value);
transcripts in transcripts/receiver_graded_<model>_set<s>.json.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("runner03", HERE / "03_api_runner.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)


def msg_receiver_graded(letter: str, game: str, rmap: list, counter: dict,
                        context_text: str) -> str:
    lines = [f"R{i + 1}: {counter[rid]}" for i, rid in enumerate(rmap)]
    return f"""Here is a text describing a situation.

TEXT {letter}:

{context_text}

In that situation there is another party on the other side: {r3.RECEIVER_ROLE[game]}.

Here are four descriptions of possible counterparts:

{chr(10).join(lines)}

TASK 1. For each description R1-R4, rate on a 0-100 scale how similar the counterpart you \
would picture on the other side of the situation in TEXT {letter} is to the person described. \
0 = nothing like the counterpart you would picture; 100 = exactly the counterpart you would \
picture. Rate each description independently; the ratings do not need to sum to any total.

TASK 2. Now distribute exactly 100 points across R1-R4 according to how strongly each \
description matches the counterpart you would picture. You may give 0 points to any description.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"ratings": {{"R1": <0-100>, "R2": <0-100>, "R3": <0-100>, "R4": <0-100>}}, \
"points": {{"R1": <int>, "R2": <int>, "R3": <int>, "R4": <int>}}}}
The points must sum to exactly 100."""


def strategic_letters(s: int) -> list:
    return [(letter, r3.CONTEXT_LABELS[s][letter]) for letter in "ABCDEFGHIJ"
            if r3.CONTEXT_LABELS[s][letter] in r3.STRATEGIC]


def run(models: list, sets: list) -> None:
    contexts = r3.load_contexts()
    _, counter = r3.load_vignettes()
    (HERE / "transcripts").mkdir(exist_ok=True)
    out_f = HERE / "recording_receiver_graded.csv"
    new = not out_f.exists()
    with open(out_f, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow("model set task context item_id value".split())
        for model in models:
            for s in sets:
                print(f"== {model} set {s}")
                conv = r3.Conversation(model, r3.MODEL_IDS[model])
                for letter, key in strategic_letters(s):
                    game = r3.STRATEGIC[key]
                    rmap = r3.COUNTERPART_ORDER[(game, s)]
                    reply = conv.send(
                        msg_receiver_graded(letter, game, rmap, counter, contexts[key]))
                    data = r3._extract_json(reply)
                    assert sum(data["points"].values()) == 100, \
                        f"points do not sum to 100 for {key} set {s}"
                    for i, rid in enumerate(rmap):
                        w.writerow([model, s, "graded", key, rid, data["ratings"][f"R{i + 1}"]])
                        w.writerow([model, s, "split", key, rid, data["points"][f"R{i + 1}"]])
                    print(f"   TEXT {letter} ({key}) done")
                (HERE / "transcripts" / f"receiver_graded_{model}_set{s}.json").write_text(
                    json.dumps(conv.history, indent=2))
                f.flush()
    print(f"wrote {out_f.name}; transcripts in transcripts/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="+", default=["claude"])
    ap.add_argument("--sets", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    if args.dry_run:
        contexts = r3.load_contexts()
        _, counter = r3.load_vignettes()
        for letter, key in strategic_letters(1):
            game = r3.STRATEGIC[key]
            rmap = r3.COUNTERPART_ORDER[(game, 1)]
            print(f"\n{'=' * 70}\n[set 1, TEXT {letter} = {key}]\n")
            print(msg_receiver_graded(letter, game, rmap, counter, contexts[key]))
        sys.exit(0)
    models = r3._available(args.models)
    if not models:
        sys.exit("no usable models (is ANTHROPIC_API_KEY set?)")
    run(models, args.sets)
