#!/usr/bin/env python3
"""Run the round-3 receiver-vignette generation (single conversation, Fable 5).

Extracts the prompt from 01_receiver_generator_prompt.md, calls the generator model
(round-2 convention: claude-fable-5; generator != rater), validates the 4x2x2 layout and
the banned-word rule, and writes receiver_vignettes_draft.json + a readable .md for
SN/AA/NG review. Freeze only after review.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR_MODEL_ID = "claude-fable-5"
MAX_TOKENS = 16000

BANNED = re.compile(
    r"\b(fair|unfair|selfish|generous|greedy|cooperat\w*|moral\w*|trust\w*|betray\w*)\b",
    re.IGNORECASE)
CLASSES = {"G": "fair_minded", "B": "indignant", "S": "own_payoff", "C": "partnership"}


def main() -> None:
    doc = (HERE / "01_receiver_generator_prompt.md").read_text()
    m = re.search(r"\*\*\*\[begin prompt[^\]]*\]\*\*\*\n(.*?)\*\*\*\[end prompt\]\*\*\*",
                  doc, flags=re.S)
    assert m, "could not extract generator prompt"
    prompt = m.group(1).strip()

    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    with client.messages.stream(
        model=GENERATOR_MODEL_ID, max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), resp.stop_reason
    reply = "".join(b.text for b in resp.content if b.type == "text")
    (HERE / "generator_transcript.json").write_text(json.dumps(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
        indent=2))

    jm = re.search(r"\{.*\}", reply, flags=re.S)
    assert jm, "no JSON in generator reply"
    data = json.loads(jm.group(0))
    vs = data["receiver_vignettes"]
    assert len(vs) == 16, f"expected 16 vignettes, got {len(vs)}"
    problems = []
    for v in vs:
        assert CLASSES[v["id"][0]] == v["class"], v["id"]
        n = len(v["text"].split())
        if not 40 <= n <= 95:
            problems.append(f"{v['id']}: {n} words")
        hits = BANNED.findall(v["text"])
        if hits:
            problems.append(f"{v['id']}: banned words {hits}")
    for cls in CLASSES:
        sub = [v for v in vs if v["id"].startswith(cls)]
        assert sorted(v["character"] for v in sub) == ["firm", "firm", "individual", "individual"]
        assert sorted(v["setting"] for v in sub) == \
            ["entrusted_proceeds", "entrusted_proceeds", "proposed_terms", "proposed_terms"]

    (HERE / "receiver_vignettes_draft.json").write_text(json.dumps(data, indent=2))
    lines = ["# Round-3 receiver vignettes - DRAFT for review (do not rate before freeze)",
             f"\nGenerator: {GENERATOR_MODEL_ID}, single conversation; layout and "
             "banned-word checks run by 02_generate_receiver_vignettes.py.\n"]
    for cls, name in CLASSES.items():
        lines.append(f"\n## Class {cls} ({name})\n")
        for v in (v for v in vs if v["id"].startswith(cls)):
            lines.append(f"**{v['id']} - {v['character']}; {v['setting']}.** {v['text']}\n")
    (HERE / "receiver_vignettes_draft.md").write_text("\n".join(lines))
    print("wrote receiver_vignettes_draft.json / .md")
    print("checks:", "ALL CLEAN" if not problems else "\n".join(problems))


if __name__ == "__main__":
    sys.exit(main())
