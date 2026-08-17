#!/usr/bin/env python3
"""Second revision of the round-3 receiver pool (SN review, 2026-07-29).

Two changes, both from the pre-pilot review recorded in post_meeting_tasks.tex \S5 and
the session notes:
  - B4: the indignant/entrusted-proceeds firm vignette showed maximally self-sacrificial
    protest (returning everything, refunding buyers). The games' Moral-bad receivers
    punish AND keep: the trustee withholds the counterpart's share for themselves. B4 is
    regenerated so the responder keeps the proceeds as compensation for the mistreatment.
  - S3, S4: remove any flavor of sharp practice or exploitation; the own-payoff class is
    neutral payoff calculation within the rules, not villainy.

Same generator convention as before (claude-fable-5, generator != rater); protagonist
names are kept so the slots stay recognizable. Updates receiver_vignettes_draft.json/.md
and appends a dated note; transcript in revise_pool_transcript.json.
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


def build_prompt(vs: dict) -> str:
    return f"""I am preparing materials for a research project on how people categorize everyday social
and economic situations. Three existing vignettes need revision. Keep each protagonist's
name, the character type (firm/business), and the setting type; change only what the
revision goal asks.

CURRENT VIGNETTES

B4 (class: indignant responder; firm; entrusted proceeds): {vs['B4']['text']}

S3 (class: own-payoff responder; firm; responding to proposed terms): {vs['S3']['text']}

S4 (class: own-payoff responder; firm; entrusted proceeds): {vs['S4']['text']}

REVISION GOALS

B4: The responder discovers the same mistreatment by the counterpart, and responds by
KEEPING the proceeds in his own hands as compensation for how he was used---punitive
toward the counterpart and materially self-serving at the same time, rather than
refunding third parties or giving everything up. The condemnation of the counterpart's
conduct must remain the clear driver; the responder may accept some risk or cost of
conflict, but he ends up keeping what he holds.

S3 and S4: Remove any hint of sharp practice, exploitation, or gaming of contracts.
The responder should come across as a neutral, unsentimental payoff calculator acting
within ordinary business terms: they take the deal or keep the share that leaves their
own firm best off, and the other side's outcome simply does not enter the decision.
No fee-stacking, no legal-loophole reasoning, no wording that invites a judgment of
misconduct.

CONSTRAINTS (unchanged from the original generation)

1. 50-80 words, third person, one named responder and one clearly identified other party.
2. Do NOT use the words "fair", "unfair", "selfish", "generous", "greedy", "cooperate",
   "cooperation", "moral", "trust", "betray", or their derivatives.
3. Do not mention experiments, studies, games, players, laboratories, or questionnaires.
4. Do not use the dollar amounts 6, 12, or 18, and do not use a tripling of any amount.
5. Keep the protagonist names (Silas, Aiko, Lorcan) and counterpart identities.

OUTPUT FORMAT

Return a single JSON object, no other text:
{{"replacements": [{{"id": "B4", "text": "..."}}, {{"id": "S3", "text": "..."}},
{{"id": "S4", "text": "..."}}]}}"""


def main() -> None:
    draft_f = HERE / "receiver_vignettes_draft.json"
    data = json.loads(draft_f.read_text())
    vs = {v["id"]: v for v in data["receiver_vignettes"]}

    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    prompt = build_prompt(vs)
    with client.messages.stream(
        model=GENERATOR_MODEL_ID, max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), resp.stop_reason
    reply = "".join(b.text for b in resp.content if b.type == "text")
    (HERE / "revise_pool_transcript.json").write_text(json.dumps(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
        indent=2))

    m = re.search(r"\{.*\}", reply, flags=re.S)
    assert m, "no JSON in reply"
    reps = json.loads(m.group(0))["replacements"]
    assert sorted(r["id"] for r in reps) == ["B4", "S3", "S4"], reps
    problems = []
    for r in reps:
        n = len(r["text"].split())
        if not 40 <= n <= 95:
            problems.append(f"{r['id']}: {n} words")
        hits = BANNED.findall(r["text"])
        if hits:
            problems.append(f"{r['id']}: banned words {hits}")
        vs[r["id"]]["text"] = r["text"]

    data["receiver_vignettes"] = [vs[v["id"]] for v in data["receiver_vignettes"]]
    draft_f.write_text(json.dumps(data, indent=2))

    classes = {"G": "fair_minded", "B": "indignant", "S": "own_payoff", "C": "partnership"}
    lines = ["# Round-3 receiver vignettes - DRAFT for review (do not rate before freeze)",
             f"\nGenerator: {GENERATOR_MODEL_ID}, single conversation; layout and "
             "banned-word checks run by 02_generate_receiver_vignettes.py.",
             "\nB2 and B4 regenerated 2026-07-29 (03_regenerate_b2_b4.py): "
             "responder-directed grievances.",
             "\nB4, S3, S4 revised 2026-07-29 (05_revise_pool.py, SN review): B4 now "
             "keeps the proceeds as punishment (matching the games' Moral-bad type); "
             "S3/S4 rewritten as neutral payoff calculation, no sharp-practice flavor.\n"]
    for cls, name in classes.items():
        lines.append(f"\n## Class {cls} ({name})\n")
        for v in (v for v in data["receiver_vignettes"] if v["id"].startswith(cls)):
            lines.append(f"**{v['id']} - {v['character']}; {v['setting']}.** {v['text']}\n")
    (HERE / "receiver_vignettes_draft.md").write_text("\n".join(lines))

    print("revised B4, S3, S4:")
    for r in reps:
        print(f"\n{r['id']}: {r['text']}")
    print("\nchecks:", "ALL CLEAN" if not problems else "\n".join(problems))


if __name__ == "__main__":
    sys.exit(main())
