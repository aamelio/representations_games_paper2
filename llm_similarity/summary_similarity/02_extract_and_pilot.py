#!/usr/bin/env python3
"""Task 7 — extract P1 story summaries from the Qualtrics exports and run the pilot.

Extraction (--extract): pulls PROLIFIC_PID, game, story, and the summary text from the
three Player-1 story surveys (between_stories: Q181, treatment = KW/LT; stories and
stories_tg: Q4) into summaries_p1.csv. Prints only aggregate counts, never text.

Pilot (--pilot): scores a stratified random sample (seed 42; 6 per game x story cell,
48 total) with the scoring prompt of 01_scoring_prompt.md, one summary per call,
randomized order, neutral ids, rater blind to game/story/condition. Writes
pilot_scores.csv and prints the variance audit (aggregates only): the Aid-Bonus contrast
per aspect (use-1 sanity) and within-cell dispersion (the gate for use 2).

Rater: claude-opus-4-8 (round-2 headline convention); needs ANTHROPIC_API_KEY.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QUALTRICS = HERE.parent.parent / "qualtrics"

MODEL_ID = "claude-opus-4-8"
MAX_TOKENS = 2000
SEED = 42
PER_CELL = 6

PROMPT = """I am running a research project on how people retell short stories about everyday \
economic situations. Below is one person's written retelling of a story. Read it and \
rate, on a 0-100 scale each, how strongly the retelling emphasizes each of the following \
three aspects of the situation. The three ratings are independent: they need not sum to \
any total, and a retelling can score high on more than one aspect or low on all three.

ASPECT F (fairness and desert): what would be right or deserved; another party's claim, \
need, or contribution; whether and how something should be divided or shared.

ASPECT O (own position): the main character's own needs, hardship, or scarcity; what \
they get to keep, use, or protect for themselves; their own material situation.

ASPECT J (joint outcomes and relationships): working together, joint contribution or \
joint success; an ongoing relationship between the parties; benefits that arise for both \
sides from the interaction.

Base your ratings only on what this retelling includes, foregrounds, or dwells on -- not \
on what you imagine the full story might contain.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"F": <0-100>, "O": <0-100>, "J": <0-100>, "dominant": "<F|O|J|balanced>", \
"note": "<one short phrase naming what the retelling foregrounds>"}}

RETELLING:

{summary}"""


def load_survey(pattern: str, qcol: str, game=None) -> pd.DataFrame:
    d = glob.glob(str(QUALTRICS / pattern))
    assert len(d) == 1, f"{pattern}: {d}"
    df = pd.read_csv(Path(d[0]) / "responses_labels.csv", skiprows=[1, 2], dtype=str)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    # story codes verified against story1tit: 2 = "The Splitting Decision" (Bonus),
    # 4 = "The Aid Program" (Aid)
    out = pd.DataFrame({
        "PROLIFIC_PID": df["PROLIFIC_PID"],
        "story": df["story"].str.strip().map({"2": "bonus", "4": "aid"}),
        "summary": df[qcol],
    })
    if game is not None:
        out["game"] = game
    else:
        out["game"] = "dg" + df["treatment"].str.strip().str.lower()
    return out


def extract() -> None:
    # UG/TG surveys have TWO columns named Q4; the first is a comprehension item and
    # pandas renames the second -- the actual story summary (verified: ~1,200 distinct
    # values, ~220 mean chars, vs 3 distinct values for the first) -- to "Q4.1".
    parts = [
        load_survey("main_collection_between_stories_*", "Q181"),
        load_survey("main_collection_stories_-*", "Q4.1", game="ug"),
        load_survey("main_collection_stories_tg*", "Q4.1", game="tg"),
    ]
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df = df.dropna(subset=["summary", "story", "PROLIFIC_PID"])
    df = df[df["summary"].str.strip().str.len() > 0]
    df["words"] = df["summary"].str.split().str.len()
    print(f"rows: {n0} -> {len(df)} after dropping empty summaries")
    print("\ncounts by game x story:")
    print(df.groupby(["game", "story"]).size().unstack(fill_value=0).to_string())
    print("\nsummary length (words) by story:")
    print(df.groupby("story")["words"].describe()[["count", "mean", "50%", "std"]]
          .round(1).to_string())
    df.to_csv(HERE / "summaries_p1.csv", index=False)
    print(f"\nwrote summaries_p1.csv ({len(df)} rows)")


def call_model(text: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    with client.messages.stream(
        model=MODEL_ID, max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        messages=[{"role": "user", "content": PROMPT.format(summary=text)}],
    ) as stream:
        resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), resp.stop_reason
    reply = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", reply, flags=re.S)
    assert m, f"no JSON in reply: {reply[:200]}"
    return json.loads(m.group(0))


def pilot() -> None:
    df = pd.read_csv(HERE / "summaries_p1.csv")
    rng = np.random.default_rng(SEED)
    take = (df.groupby(["game", "story"], group_keys=False)
              .apply(lambda g: g.sample(n=min(PER_CELL, len(g)), random_state=SEED)))
    take = take.sample(frac=1, random_state=SEED + 1).reset_index(drop=True)  # shuffle order
    print(f"pilot sample: {len(take)} summaries "
          f"({take.groupby(['game', 'story']).size().min()}-"
          f"{take.groupby(['game', 'story']).size().max()} per cell)")
    rows = []
    for i, r in take.iterrows():
        s = call_model(r["summary"])
        rows.append(dict(PROLIFIC_PID=r["PROLIFIC_PID"], game=r["game"], story=r["story"],
                         F=s["F"], O=s["O"], J=s["J"],
                         dominant=s["dominant"], note=s["note"]))
        print(f"  {i + 1}/{len(take)} scored")
    sc = pd.DataFrame(rows)
    sc.to_csv(HERE / "pilot_scores.csv", index=False)

    print("\n=== use-1 sanity: mean scores by story (pooled over games) ===")
    print(sc.groupby("story")[["F", "O", "J"]].mean().round(1).to_string())
    print("\nby game x story:")
    print(sc.groupby(["game", "story"])[["F", "O", "J"]].mean().round(1).to_string())
    print("\ndominant shares by story:")
    print(sc.groupby("story")["dominant"].value_counts(normalize=True).round(2).to_string())
    print("\n=== use-2 gate: within-cell dispersion (SD of scores) ===")
    print(sc.groupby(["game", "story"])[["F", "O", "J"]].std().round(1).to_string())
    print("\nwrote pilot_scores.csv (notes saved to file, not printed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if not (args.extract or args.pilot):
        ap.error("pass --extract and/or --pilot")
    if args.extract:
        extract()
    if args.pilot:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set")
        pilot()
