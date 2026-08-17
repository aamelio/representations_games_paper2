#!/usr/bin/env python3
"""Regenerate receiver vignettes B2 and B4 (round-3 pool, review point of 2026-07-28).

Reason: in the first draft, B2 and B4 punish misconduct aimed at THIRD PARTIES (a
defrauded buyer; dispossessed tenants), whereas the games' Moral-bad receivers condemn
conduct aimed at THEMSELVES (the one-sided terms they face). The two slots are re-drawn
with responder-directed grievances; B1 and B3 and all other classes are untouched.

Single fresh conversation with the generator model (round-2 convention: claude-fable-5;
generator != rater). Updates receiver_vignettes_draft.json/.md in place and appends a
regeneration note; transcript in regen_b2_b4_transcript.json.
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

USED_NAMES = (
    # AA's 48 sender protagonists
    "Elena Malik Ruth Jun Aditya Beatriz Calvin Dalia Farah Gideon Hyejin Isaac Jocelyn "
    "Karim Leona Mateo Noura Omar Paloma Quentin Rina Stefan Talia Ugo Valerie Waleed "
    "Ximena Yusuf Zora Anders Bruna Chen Deepa Emil Fatima Gabriel Helena Idris Johanna "
    "Kwame Lidia Musa Noemi Pavel Rasha Soren Tamara Viktor "
    # round-3 receiver draft protagonists and named counterparts
    "Oskar Denny Marisol Aldo Ingrid Dmitri Lena Yara Piet Hassan Roy Petra Nnamdi "
    "Wendy Carol Bram Aiko Lorcan Sylvie Marco Priya Devi Tobias Colette"
).split()


def build_prompt(current_b: list) -> str:
    b_texts = "\n\n".join(f"{v['id']} ({v['character']}; {v['setting']}): {v['text']}"
                          for v in current_b)
    return f"""I am preparing materials for a research project on how people categorize everyday social
and economic situations. An earlier conversation produced four vignettes for a class of
"indignant responders", defined as follows:

CLASS B (indignant responder): the responder reads the other side's conduct as wrong —
one-sided terms, an exploitative arrangement, someone taking advantage — and responds by
refusing the terms or withholding everything, even at a real cost to themselves. The
point of the response is to condemn and punish the conduct, not to gain.

Here are the four current vignettes:

{b_texts}

Two of them (B2 and B4) have a flaw: the misconduct they punish is aimed at a THIRD
PARTY (a deceived buyer; dispossessed tenants). For this project, the condemned conduct
must be aimed AT THE RESPONDER: the responder discovers that the very party whose
arrangement they are part of has treated THEM one-sidedly, deceptively, or
exploitatively, and withholds everything in protest, at a real cost to themselves.

Write REPLACEMENTS for B2 and B4 only, with responder-directed grievances:

- B2: an individual person as responder; the setting involves money or goods the other
  party entrusted, advanced, or handed over first, from which the responder was to keep
  or pass on a share.
- B4: a firm, shop, or business owner as responder; same kind of setting (entrusted or
  advanced resources), in a market context.

CONSTRAINTS (same as the original generation)

1. 50-80 words, third person, one named responder as protagonist and one clearly
   identified other party.
2. Do NOT use the words "fair", "unfair", "selfish", "generous", "greedy", "cooperate",
   "cooperation", "moral", "trust", "betray", or their derivatives: the class must be
   conveyed by what the responder does and weighs, not by naming it.
3. Do not mention experiments, studies, games, players, laboratories, or questionnaires.
4. Do not use the dollar amounts 6, 12, or 18, and do not use a tripling of any amount.
5. Do not reuse any of these names (already used in the project): {" ".join(USED_NAMES)}.
6. Do not duplicate the scenarios of B1 and B3 above; vary resource, relationship, and
   setting.

OUTPUT FORMAT

Return a single JSON object, no other text:
{{"replacements": [{{"id": "B2", "class": "indignant", "character": "individual",
"setting": "entrusted_proceeds", "text": "<the vignette>"}}, {{"id": "B4", "class":
"indignant", "character": "firm", "setting": "entrusted_proceeds", "text": "<the
vignette>"}}]}}"""


def main() -> None:
    draft_f = HERE / "receiver_vignettes_draft.json"
    data = json.loads(draft_f.read_text())
    vs = {v["id"]: v for v in data["receiver_vignettes"]}
    current_b = [vs[i] for i in ["B1", "B2", "B3", "B4"]]

    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    prompt = build_prompt(current_b)
    with client.messages.stream(
        model=GENERATOR_MODEL_ID, max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), resp.stop_reason
    reply = "".join(b.text for b in resp.content if b.type == "text")
    (HERE / "regen_b2_b4_transcript.json").write_text(json.dumps(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
        indent=2))

    jm = re.search(r"\{.*\}", reply, flags=re.S)
    assert jm, "no JSON in reply"
    reps = json.loads(jm.group(0))["replacements"]
    assert [r["id"] for r in reps] == ["B2", "B4"], reps
    problems = []
    for r in reps:
        n = len(r["text"].split())
        if not 40 <= n <= 95:
            problems.append(f"{r['id']}: {n} words")
        hits = BANNED.findall(r["text"])
        if hits:
            problems.append(f"{r['id']}: banned words {hits}")
        for name in USED_NAMES:
            if re.search(rf"\b{name}\b", r["text"]):
                problems.append(f"{r['id']}: reused name {name}")
        vs[r["id"]] = r

    data["receiver_vignettes"] = [vs[v["id"]] for v in data["receiver_vignettes"]]
    draft_f.write_text(json.dumps(data, indent=2))

    classes = {"G": "fair_minded", "B": "indignant", "S": "own_payoff", "C": "partnership"}
    lines = ["# Round-3 receiver vignettes - DRAFT for review (do not rate before freeze)",
             f"\nGenerator: {GENERATOR_MODEL_ID}, single conversation; layout and "
             "banned-word checks run by 02_generate_receiver_vignettes.py.",
             "\nB2 and B4 regenerated 2026-07-29 (03_regenerate_b2_b4.py): the first "
             "drafts punished third-party misconduct, whereas the games' Moral-bad "
             "receivers condemn conduct aimed at themselves; replacements have "
             "responder-directed grievances. B1/B3 and all other classes unchanged.\n"]
    for cls, name in classes.items():
        lines.append(f"\n## Class {cls} ({name})\n")
        for v in (v for v in data["receiver_vignettes"] if v["id"].startswith(cls)):
            lines.append(f"**{v['id']} - {v['character']}; {v['setting']}.** {v['text']}\n")
    (HERE / "receiver_vignettes_draft.md").write_text("\n".join(lines))

    print("replaced B2 and B4:")
    for r in reps:
        print(f"\n{r['id']}: {r['text']}")
    print("\nchecks:", "ALL CLEAN" if not problems else "\n".join(problems))


if __name__ == "__main__":
    sys.exit(main())
