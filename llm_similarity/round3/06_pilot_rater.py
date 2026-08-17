#!/usr/bin/env python3
"""Round-3 PILOT rating run --- draft pool, PRE-FREEZE (2026-07-29).

Rates the ten contexts of the design against the draft 48-sender + 16-receiver pool, to
give AA and NG concrete draft results for the design discussion. This is explicitly NOT
the headline measurement: the pool is not frozen, and the headline run will be redone
from scratch after the coauthors review and freeze the vignettes.

Design mirrors round 2 (02_rater_prompts.md / 03_api_runner.py): headline rater
(Opus 4.8, reasoning on), three conversations with independently permuted neutral labels
(V1--V48 for sender vignettes, R1--R16 for receiver vignettes, TEXT A--J for contexts;
permutations drawn with seed 20260729 and stored in pilot_mappings.json); per context, a
graded 0--100 rating for every sender vignette plus a 100-point retrieval split; for the
four strategic contexts, a graded 0--100 rating for every receiver vignette plus a
100-point split.

Usage:
  python3 06_pilot_rater.py --dry-run     # print set-1 message sizes, no API calls
  python3 06_pilot_rater.py               # run (needs ANTHROPIC_API_KEY)

Outputs: pilot_recording.csv, pilot_retrieval.csv, pilot_receiver.csv,
pilot_mappings.json, transcripts in pilot_transcripts/.
"""

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "round2"

spec = importlib.util.spec_from_file_location("runner03", ROUND2 / "03_api_runner.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)

SEED = 20260729
MODEL = "claude"  # provider key in 03_api_runner.MODEL_IDS -> claude-opus-4-8
CONTEXTS = r2.CONTEXT_KEYS          # BONUS AID C-KW C-LT C-UG C-TG M-KW M-LT M-UG M-TG
STRATEGIC = r2.STRATEGIC            # {C-UG: UG, M-UG: UG, C-TG: TG, M-TG: TG}


def load_sender_pool() -> dict:
    text = (HERE.parent / "vignettes_6rep_draft.md").read_text()
    items = re.findall(r"^\*\*([A-Z]{2}\d)\s*[—–-]+\s*[^.]+\.\*\*\s*(.+)$", text, re.M)
    assert len(items) == 48, f"expected 48 sender vignettes, found {len(items)}"
    return dict(items)


def load_receiver_pool() -> dict:
    data = json.loads((HERE / "receiver_vignettes_draft.json").read_text())
    vs = {v["id"]: v["text"] for v in data["receiver_vignettes"]}
    assert len(vs) == 16
    return vs


def draw_mappings() -> dict:
    rng = random.Random(SEED)
    sender_ids = sorted(load_sender_pool())
    receiver_ids = sorted(load_receiver_pool())
    maps = {}
    for s in [1, 2, 3]:
        ctx = list(CONTEXTS)
        rng.shuffle(ctx)
        sv = list(sender_ids)
        rng.shuffle(sv)
        rv = list(receiver_ids)
        rng.shuffle(rv)
        maps[s] = {"contexts": dict(zip("ABCDEFGHIJ", ctx)),
                   "sender": sv, "receiver": rv}
    return maps


def msg0(vmap: list, sender: dict) -> str:
    lines = [f"V{i + 1}: {sender[vid]}" for i, vid in enumerate(vmap)]
    return ("I am running a research project on how people perceive the similarity "
            "between everyday situations. I will first give you 48 short vignettes, each "
            "describing a situation (labeled V1 to V48). In later messages I will show "
            "you other texts and ask you to rate their similarity to these vignettes. "
            'For now, read the vignettes and reply only: "Ready."\n\n' + "\n\n".join(lines))


def msg_context(letter: str, text: str) -> str:
    return f"""Here is a text describing a situation.

TEXT {letter}:

{text}

TASK 1. For each vignette V1-V48, rate on a 0-100 scale how similar the situation in TEXT \
{letter} is to the situation in the vignette, considering the situation as a whole: what is \
at stake, the relationship between the parties, the setting, and the structure of the \
decision each party faces. 0 = the two situations have nothing in common; 100 = they are \
essentially the same situation.

TASK 2. Now imagine a person in the situation of TEXT {letter}, asking themselves: "What \
kind of situation is this? What does it remind me of?" Distribute exactly 100 points across \
the 48 vignettes according to how strongly each would come to mind as a similar experience. \
You may give 0 points to any vignette.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"ratings": {{"V1": <0-100>, ..., "V48": <0-100>}}, "retrieval_points": {{"V1": <int>, ..., "V48": <int>}}}}
The retrieval points must sum to exactly 100."""


def msg_receiver(letter: str, game: str, rmap: list, receiver: dict) -> str:
    lines = [f"R{i + 1}: {receiver[rid]}" for i, rid in enumerate(rmap)]
    return f"""Consider again TEXT {letter}. In that situation there is another party on the other \
side: {r2.RECEIVER_ROLE[game]}.

Here are sixteen descriptions of possible counterparts:

{chr(10).join(lines)}

TASK 1. For each description R1-R16, rate on a 0-100 scale how similar the counterpart you \
would picture on the other side of the situation in TEXT {letter} is to the person or firm \
described. Rate each description independently; the ratings need not sum to any total.

TASK 2. Now distribute exactly 100 points across R1-R16 according to how strongly each \
description matches the counterpart you would picture. You may give 0 points to any \
description.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"ratings": {{"R1": <0-100>, ..., "R16": <0-100>}}, "points": {{"R1": <int>, ..., "R16": <int>}}}}
The points must sum to exactly 100."""


def send_checked(conv, message: str, points_key: str, tag) -> dict:
    """Send a rating message; if the 100-point allocation misses 100, ask once for a
    corrected JSON; if it still misses, keep the reply and log (the analysis normalizes
    by the actual sum)."""
    data = r2._extract_json(conv.send(message))
    total = sum(data[points_key].values())
    if total != 100:
        print(f"   [{tag}] {points_key} sum to {total}; asking for a correction", flush=True)
        data = r2._extract_json(conv.send(
            f"Your {points_key} sum to {total}, not 100. Please resend ONLY the corrected "
            "JSON object, adjusting the points so they sum to exactly 100 and keeping "
            "everything else as close as possible to your previous answer."))
        total = sum(data[points_key].values())
        if total != 100:
            print(f"   [{tag}] still {total} after correction; keeping (normalized later)",
                  flush=True)
    return data


def run(sets: list) -> None:
    sender, receiver = load_sender_pool(), load_receiver_pool()
    contexts = r2.load_contexts()
    maps = draw_mappings()
    (HERE / "pilot_mappings.json").write_text(json.dumps(maps, indent=2))
    (HERE / "pilot_transcripts").mkdir(exist_ok=True)

    rec_f = HERE / "pilot_recording.csv"
    ret_f = HERE / "pilot_retrieval.csv"
    rcv_f = HERE / "pilot_receiver.csv"
    new = not rec_f.exists()
    with open(rec_f, "a", newline="") as f1, open(ret_f, "a", newline="") as f2, \
         open(rcv_f, "a", newline="") as f3:
        w1, w2, w3 = csv.writer(f1), csv.writer(f2), csv.writer(f3)
        if new:
            w1.writerow("set context vignette_id rating".split())
            w2.writerow("set context vignette_id points".split())
            w3.writerow("set task context receiver_id value".split())
        for s in sets:
            print(f"== pilot set {s}", flush=True)
            m = maps[s]
            conv = r2.Conversation(MODEL, r2.MODEL_IDS[MODEL])
            conv.send(msg0(m["sender"], sender))
            for letter in "ABCDEFGHIJ":
                key = m["contexts"][letter]
                data = send_checked(conv, msg_context(letter, contexts[key]),
                                    "retrieval_points", (s, key))
                for i, vid in enumerate(m["sender"]):
                    w1.writerow([s, key, vid, data["ratings"][f"V{i + 1}"]])
                    w2.writerow([s, key, vid, data["retrieval_points"][f"V{i + 1}"]])
                print(f"   TEXT {letter} ({key}) done", flush=True)
            for letter in "ABCDEFGHIJ":
                key = m["contexts"][letter]
                if key in STRATEGIC:
                    data = send_checked(
                        conv, msg_receiver(letter, STRATEGIC[key], m["receiver"], receiver),
                        "points", (s, key))
                    for i, rid in enumerate(m["receiver"]):
                        w3.writerow([s, "graded", key, rid, data["ratings"][f"R{i + 1}"]])
                        w3.writerow([s, "split", key, rid, data["points"][f"R{i + 1}"]])
                    print(f"   receiver module {key} done", flush=True)
            (HERE / "pilot_transcripts" / f"rater_set{s}.json").write_text(
                json.dumps(conv.history, indent=2))
            f1.flush(); f2.flush(); f3.flush()
    print("pilot run complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sets", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    if args.dry_run:
        sender, receiver = load_sender_pool(), load_receiver_pool()
        maps = draw_mappings()
        m0 = msg0(maps[1]["sender"], sender)
        ctx = r2.load_contexts()
        mc = msg_context("A", ctx[maps[1]["contexts"]["A"]])
        mr = msg_receiver("A", "UG", maps[1]["receiver"], receiver)
        print(f"set-1 context order: {maps[1]['contexts']}")
        print(f"message sizes (words): msg0 {len(m0.split())}, "
              f"context {len(mc.split())}, receiver {len(mr.split())}")
        sys.exit(0)
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    run(args.sets)
