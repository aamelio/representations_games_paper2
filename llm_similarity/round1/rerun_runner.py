#!/usr/bin/env python3
"""Round-1 rerun with the neutral similarity prompt (NG meeting 2026-07-23, item 3).

Reruns the May round-1 protocol with Prompt 1 replaced by the neutral version of
rerun_neutral_prompt.md (AA's draft + agreed amendments); Prompts 2 and 3 are verbatim
from the May doc. Materials (game/story texts) are the May texts, frozen in
rerun_materials.json; the three triplet label permutations are the May mappings.

Claude arm runs on the SAME model as the May round (Opus 4.7), so the comparison
isolates the prompt change. GPT/Gemini arms: add ids/keys and extend _call.

API output format: per-pair justifications first (reasoning before numbers, as in the
May order), then one JSON object in place of the summary markdown table.

Usage:
  python3 rerun_runner.py --dry-run          # print triplet-1 prompts, no API calls
  python3 rerun_runner.py                    # run 3 triplets (needs ANTHROPIC_API_KEY)

Output: rerun_recording.csv, rerun_splits.csv (canonical game/story names);
transcripts in transcripts_rerun/.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODEL_ID = "claude-opus-4-7"  # May round-1 Claude arm: same model, new prompt
MAX_TOKENS = 32000

TRIPLETS = {
    1: dict(games=["DG-KW", "DG-LT", "UG", "TG"], stories=["Bonus", "Aid"]),
    2: dict(games=["TG", "UG", "DG-LT", "DG-KW"], stories=["Aid", "Bonus"]),
    3: dict(games=["DG-LT", "DG-KW", "TG", "UG"], stories=["Aid", "Bonus"]),
}
GAME_LABELS = ["A", "B", "C", "D"]
STORY_LABELS = ["Story 1", "Story 2"]


def load_materials() -> dict:
    return json.loads((HERE / "rerun_materials.json").read_text())


def prompt1(t: int, mat: dict) -> str:
    games = TRIPLETS[t]["games"]
    stories = TRIPLETS[t]["stories"]
    blocks = []
    for lab, g in zip(GAME_LABELS, games):
        blocks.append(f"--- GAME {lab} ---\n\n{mat[g]}")
    for lab, s in zip(STORY_LABELS, stories):
        blocks.append(f"--- {lab.upper()} ---\n\n{mat[s]}")
    return f"""I am running a research project comparing real-world stories with decision situations. \
I will give you descriptions of four games and two short stories.

INSTRUCTIONS

Below are four games (A, B, C, D) and two stories (Story 1, Story 2).

For each of the 8 story x game pairs, rate how similar the two situations are in their \
underlying decision structure on a 0-100 scale:

0 = the situations have no meaningful similarity in their underlying decision structure

100 = the situations have essentially the same underlying decision structure

Assess each pair independently and base your judgment only on the descriptions provided. \
Use your own judgment to decide which aspects of the situations are relevant and how much \
weight to give them. Apply the same standard across all eight pairs. Equal ratings are \
allowed, and the ratings do not need to sum to any particular total.

OUTPUT FORMAT

1. For each of the 8 pairs, give the rating and a 2-3 sentence justification identifying \
the main considerations behind the rating.

2. At the end, output a single JSON object with the 8 ratings, exactly in this form:
{{"ratings": {{"A": {{"Story 1": <0-100>, "Story 2": <0-100>}}, "B": {{...}}, "C": {{...}}, "D": {{...}}}}}}

3. After the JSON, briefly identify the most and least similar game or games for each \
story and explain why. Allow for ties.

{chr(10).join(blocks)}"""


def prompt2() -> str:
    return """Now decompose your similarity judgments along explicit structural dimensions. \
For each of the four games (A, B, C, D) and each of the two stories (Story 1, Story 2) you \
analyzed above, specify the value on each of the following dimensions:

1. Pie structure: fixed-sum, or expandable through one player's action?

2. Player 1's initial endowment: full pie, or partial (recipient also gets a positive amount)?

3. Player 2's role: passive recipient, accept/reject responder, or reciprocator with a \
continuous choice?

4. Source of Player 1's endowment: earned through effort, awarded by selection or luck, \
produced through interaction, or unspecified?

5. Strategic structure: unilateral allocation, proposer-responder with veto, or sequential \
interaction with reciprocity?

OUTPUT FORMAT

1. Produce a table with six columns (A, B, C, D, Story 1, Story 2) and five rows (one per \
dimension). Fill each cell with the value on that dimension for that game or story.

2. Then answer these three questions explicitly:

(a) Which dimensions account for the largest similarity gaps in your Pass 1 ratings?

(b) Is any single dimension responsible for distinguishing one of the four games from the \
other three in its similarity to either story? If yes, name the dimension and the game.

(c) Does the answer to (b) differ across the two stories, or is it the same?"""


def prompt3() -> str:
    return """Now consider the reverse retrieval direction. For each of the four games (A, B, C, D), \
imagine a participant who has just read the game instructions for the first time and is \
asked: "Does this situation remind you of any real-world experience?"

Of the two stories (Story 1 and Story 2), which is more likely to be retrieved as a similar \
past experience cued by each game?

For each game, distribute 100 points between Story 1 and Story 2 to indicate relative \
retrieval likelihood. For example, 70/30 means Story 1 is much more likely to come to mind \
than Story 2 when cued by that game; 50/50 means equal likelihood.

OUTPUT FORMAT

1. For each game, give the 100-point split (Story 1 / Story 2) and a 2-3 sentence \
justification.

2. At the end, output a single JSON object with the splits, exactly in this form:
{{"splits": {{"A": {{"Story 1": <int>, "Story 2": <int>}}, "B": {{...}}, "C": {{...}}, "D": {{...}}}}}}
Each game's two numbers must sum to 100.

3. After the JSON, comment briefly on whether the relative-retrieval ranking across games \
matches your Pass 1 absolute similarity ratings, or differs (and if so, where and why).""".replace("{{", "{").replace("}}", "}")


def call_claude(history: list) -> str:
    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    kwargs = dict(model=MODEL_ID, max_tokens=MAX_TOKENS, messages=history)
    try:
        with client.messages.stream(thinking={"type": "adaptive"},
                                    output_config={"effort": "high"}, **kwargs) as stream:
            resp = stream.get_final_message()
    except (TypeError, anthropic.BadRequestError):
        # older tier without adaptive thinking / output_config: fixed thinking budget
        with client.messages.stream(thinking={"type": "enabled", "budget_tokens": 10000},
                                    **kwargs) as stream:
            resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), \
        f"unexpected stop_reason={resp.stop_reason}"
    return "".join(b.text for b in resp.content if b.type == "text")


def send(history: list, text: str) -> str:
    history.append({"role": "user", "content": text})
    reply = call_claude(history)
    history.append({"role": "assistant", "content": reply})
    return reply


def tail_json(reply: str, key: str) -> dict:
    """Parse the LAST JSON object containing `key` (justifications may precede it)."""
    starts = [m.start() for m in re.finditer(r"\{\s*\"" + key + r"\"", reply)]
    assert starts, f"no JSON object with key '{key}' found:\n{reply[:400]}"
    s = starts[-1]
    depth = 0
    for i in range(s, len(reply)):
        if reply[i] == "{":
            depth += 1
        elif reply[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(reply[s:i + 1])
    raise AssertionError("unbalanced JSON braces in reply")


def run(triplets: list) -> None:
    mat = load_materials()
    (HERE / "transcripts_rerun").mkdir(exist_ok=True)
    rec_f, spl_f = HERE / "rerun_recording.csv", HERE / "rerun_splits.csv"
    new_r, new_s = not rec_f.exists(), not spl_f.exists()
    with open(rec_f, "a", newline="") as fr, open(spl_f, "a", newline="") as fs:
        wr, ws = csv.writer(fr), csv.writer(fs)
        if new_r:
            wr.writerow("model triplet game story rating".split())
        if new_s:
            ws.writerow("model triplet game story points".split())
        for t in triplets:
            print(f"== {MODEL_ID} triplet {t}")
            games, stories = TRIPLETS[t]["games"], TRIPLETS[t]["stories"]
            history = []
            r1 = send(history, prompt1(t, mat))
            ratings = tail_json(r1, "ratings")["ratings"]
            for lab, g in zip(GAME_LABELS, games):
                for slab, s in zip(STORY_LABELS, stories):
                    wr.writerow([MODEL_ID, t, g, s, ratings[lab][slab]])
            print("   prompt 1 done")
            send(history, prompt2())
            print("   prompt 2 done")
            r3_ = send(history, prompt3())
            splits = tail_json(r3_, "splits")["splits"]
            for lab, g in zip(GAME_LABELS, games):
                row = splits[lab]
                assert row[STORY_LABELS[0]] + row[STORY_LABELS[1]] == 100, \
                    f"split does not sum to 100 for game {g}, triplet {t}"
                for slab, s in zip(STORY_LABELS, stories):
                    ws.writerow([MODEL_ID, t, g, s, row[slab]])
            print("   prompt 3 done")
            (HERE / "transcripts_rerun" / f"rater_claude_triplet{t}.json").write_text(
                json.dumps(history, indent=2))
            fr.flush(); fs.flush()
    print(f"wrote {rec_f.name}, {spl_f.name}; transcripts in transcripts_rerun/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--triplets", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    if args.dry_run:
        mat = load_materials()
        print(prompt1(1, mat))
        print("\n" + "=" * 70 + "\n")
        print(prompt3())
        sys.exit(0)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    run(args.triplets)
