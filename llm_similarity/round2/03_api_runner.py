#!/usr/bin/env python3
"""Similarity round 2 — API runner.

Runs the vignette-generation conversation and/or the 9 rater conversations
(3 models x 3 permutation sets) defined in 01_generator_prompt.md and
02_rater_prompts.md, and writes the recording CSVs.

Usage:
  python3 03_api_runner.py --generate            # run generator, write vignettes.json (review, then freeze into vignettes.md)
  python3 03_api_runner.py --rate                # run all 9 rater conversations (needs vignettes.json)
  python3 03_api_runner.py --rate --models claude --sets 1   # subset, for piloting

Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (env vars). Models missing a key are
skipped with a warning. Raw transcripts land in transcripts/, parsed numbers in
recording_round2.csv / recording_splits_round2.csv (same schemas as the *_template.csv files).

API mode asks for JSON output instead of markdown tables (content identical to the manual
prompts; only the OUTPUT FORMAT paragraph differs). Update MODEL_IDS before running.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- configuration
MODEL_IDS = {
    # Protocol 2026-07-19 (SN): raters = flagship generally-available tier of each provider, with
    # reasoning enabled (claude adapter passes adaptive thinking + effort high explicitly, since
    # the API default is thinking-off).
    "claude": "claude-opus-4-8",
    "claude_fable": "claude-fable-5",  # within-provider robustness rater (04 keeps it out of the headline)
    "gpt": "UPDATE-ME (OpenAI reasoning-tier model id)",
    "gemini": "UPDATE-ME (Gemini Pro model id)",
}
GENERATOR_MODEL = "claude"  # provider key (for API-key lookup)
GENERATOR_MODEL_ID = "claude-fable-5"  # most capable tier for the constrained one-shot; generator != rater
MAX_TOKENS = 32000  # headroom for thinking + JSON; claude adapter streams, so no HTTP-timeout risk

CONTEXT_KEYS = ["BONUS", "AID", "C-KW", "C-LT", "C-UG", "C-TG", "M-KW", "M-LT", "M-UG", "M-TG"]
STRATEGIC = {"C-UG": "UG", "M-UG": "UG", "C-TG": "TG", "M-TG": "TG"}

CONTEXT_LABELS = {  # TEXT letter -> canonical key, per set (02_rater_prompts.md, private tables)
    1: dict(zip("ABCDEFGHIJ", ["C-UG", "BONUS", "M-KW", "C-TG", "AID", "M-UG", "C-KW", "M-TG", "C-LT", "M-LT"])),
    2: dict(zip("ABCDEFGHIJ", ["M-TG", "C-LT", "AID", "M-UG", "C-KW", "BONUS", "M-LT", "C-UG", "C-TG", "M-KW"])),
    3: dict(zip("ABCDEFGHIJ", ["AID", "M-LT", "C-TG", "C-KW", "M-TG", "C-UG", "BONUS", "M-KW", "C-LT", "M-UG"])),
    # set 4: substitute permutation for the Fable-5 robustness rater's third conversation
    # (random.Random(20260719), drawn 2026-07-19): set 3's opening message was declined three
    # times by that model's safety filter (false positive, order-dependent); permutations are
    # nuisance parameters, so a fresh draw preserves the protocol. Opus headline = sets 1-3.
    4: dict(zip("ABCDEFGHIJ", ["M-TG", "M-KW", "BONUS", "C-KW", "C-LT", "M-LT", "AID", "C-TG", "M-UG", "C-UG"])),
    # set 5: second substitute for the same conversation (random.Random(20260720), successor
    # seed, identical draw procedure — verified to reproduce set 4 from 20260719). Drawn
    # 2026-07-19 after set 4's opening message was also declined (5th refusal) while a same-day
    # probe of set 1's opening message completed cleanly — i.e., the filter's false positive is
    # ordering-specific, not account state. Selection caveat: the third conversation's
    # permutation is thereby conditioned on passing the filter; disclosed in the appendix.
    5: dict(zip("ABCDEFGHIJ", ["C-UG", "AID", "BONUS", "M-LT", "C-LT", "M-KW", "M-TG", "M-UG", "C-TG", "C-KW"])),
    # set 6: the substitute permutation that completed the third Fable-5 conversation
    # (2026-07-19). Found by a pre-committed successor-seed hunt (probe = opening message only,
    # one call per seed, stop at first acceptance): set 5 (20260720) refused; 20260721-24
    # refused; 20260725 accepted. The filter then refused the IDENTICAL accepted message on the
    # first full-run attempt and accepted it on the next, so its false positives are stochastic
    # per request, not a deterministic function of the ordering — which makes retrying a fixed
    # permutation selection-free. Full run completed on retry attempt 1 after cleanup; refusal
    # tally and the disclosure are in the appendix and ng_comments_tracker.md.
    6: dict(zip("ABCDEFGHIJ", ["AID", "M-KW", "M-LT", "C-TG", "C-LT", "M-UG", "M-TG", "BONUS", "C-KW", "C-UG"])),
}
VIGNETTE_ORDER = {  # V1..V24 -> canonical vignette id, per set
    1: "S3 C7 M1 C2 S8 M6 C4 S1 M8 C1 S5 M3 C8 S2 M7 C5 S6 M2 C3 S7 M4 C6 S4 M5".split(),
    2: "M4 S6 C1 M2 C8 S3 M7 C5 S1 M5 C3 S8 M1 C6 S4 M8 C2 S7 M3 C7 S2 M6 C4 S5".split(),
    3: "C6 M5 S2 C4 M8 S7 C1 M3 S5 C8 M1 S4 C3 M6 S8 C7 M2 S6 C5 M7 S3 C2 M4 S1".split(),
    4: "S1 C5 S7 M2 M4 S6 S5 C4 M6 C2 C1 M1 C7 M3 M8 M7 S4 C6 C8 S2 C3 M5 S3 S8".split(),
    5: "C3 S6 S7 M8 S5 C1 C7 M2 S8 C4 C2 M6 C8 C6 C5 S1 S4 M1 S2 S3 M3 M5 M4 M7".split(),
    6: "M3 C1 S6 S7 M4 M1 S1 M6 C4 C5 S3 C7 M5 S4 C6 S5 M8 S8 C3 M7 C2 C8 S2 M2".split(),
}
COUNTERPART_ORDER = {  # R1..R4 per set, per game module
    ("UG", 1): ["UR2", "UA1", "UR1", "UA2"],
    ("UG", 2): ["UA2", "UR1", "UA1", "UR2"],
    ("UG", 3): ["UR1", "UA2", "UR2", "UA1"],
    ("UG", 4): ["UA2", "UR2", "UR1", "UA1"],
    ("UG", 5): ["UR2", "UR1", "UA2", "UA1"],
    ("UG", 6): ["UA1", "UA2", "UR1", "UR2"],
    ("TG", 1): ["TK1", "TR2", "TR1", "TK2"],
    ("TG", 2): ["TR1", "TK2", "TK1", "TR2"],
    ("TG", 3): ["TK2", "TR1", "TR2", "TK1"],
    ("TG", 4): ["TR1", "TK1", "TR2", "TK2"],
    ("TG", 5): ["TR2", "TK2", "TR1", "TK1"],
    ("TG", 6): ["TR2", "TK2", "TR1", "TK1"],
}
RECEIVER_ROLE = {
    "UG": "the person who sees the proposed terms and decides whether to accept or reject them",
    "TG": "the person who receives what is sent and decides how much to give back",
}


# ---------------------------------------------------------------- provider adapters
class Conversation:
    """Multi-turn conversation against one provider; keeps message history."""

    def __init__(self, provider, model_id):
        self.provider, self.model_id, self.history = provider, model_id, []

    def send(self, text):
        self.history.append({"role": "user", "content": text})
        reply = _call(self.provider, self.model_id, self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply


def _call(provider, model_id, history):
    if provider in ("claude", "claude_fable"):
        import anthropic
        client = anthropic.Anthropic(max_retries=8)  # low-tier rate limits: retry through 429 bursts
        with client.messages.stream(
            model=model_id, max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"}, output_config={"effort": "high"},
            messages=history,
        ) as stream:
            resp = stream.get_final_message()
        assert resp.stop_reason in ("end_turn", "stop_sequence"), \
            f"unexpected stop_reason={resp.stop_reason} (refusal/truncation would corrupt the protocol)"
        return "".join(b.text for b in resp.content if b.type == "text")
    if provider == "gpt":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=model_id, messages=history)
        return resp.choices[0].message.content
    if provider == "gemini":
        from google import genai
        client = genai.Client()
        chat_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history[:-1]
        ]
        chat = client.chats.create(model=model_id, history=chat_history)
        return chat.send_message(history[-1]["content"]).text
    raise ValueError(provider)


def _available(models):
    keys = {"claude": "ANTHROPIC_API_KEY", "claude_fable": "ANTHROPIC_API_KEY",
            "gpt": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    out = []
    for m in models:
        if "UPDATE-ME" in MODEL_IDS[m]:
            print(f"[skip] {m}: MODEL_IDS placeholder not updated")
        elif not os.environ.get(keys[m]):
            print(f"[skip] {m}: {keys[m]} not set")
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------- materials
def load_contexts():
    """Parse contexts.md into {key: text}."""
    raw = (HERE / "contexts.md").read_text()
    parts = re.split(r"^## (\S+) ", raw, flags=re.M)[1:]
    ctx = {parts[i]: parts[i + 1].split("\n", 1)[1].strip() for i in range(0, len(parts), 2)}
    missing = [k for k in CONTEXT_KEYS if k not in ctx]
    assert not missing, f"contexts.md missing {missing}"
    return ctx


def load_vignettes():
    f = HERE / "vignettes.json"
    assert f.exists(), "vignettes.json missing - run --generate first, review, then --rate"
    data = json.loads(f.read_text())
    sender = {v["id"]: v["text"] for v in data["sender_vignettes"]}
    counter = {v["id"]: v["text"] for v in data["counterpart_vignettes"]}
    assert len(sender) == 24 and len(counter) == 8
    return sender, counter


def _extract_json(reply):
    m = re.search(r"\{.*\}", reply, flags=re.S)
    assert m, f"no JSON object found in reply:\n{reply[:500]}"
    return json.loads(m.group(0))


# ---------------------------------------------------------------- generation
def run_generate():
    prompt_doc = (HERE / "01_generator_prompt.md").read_text()
    m = re.search(r"\*\*\*\[begin prompt[^\]]*\]\*\*\*\n(.*?)\*\*\*\[end prompt\]\*\*\*",
                  prompt_doc, flags=re.S)
    assert m, "could not extract generator prompt"
    provider = _available([GENERATOR_MODEL])
    assert provider, "generator model unavailable"
    conv = Conversation(GENERATOR_MODEL, GENERATOR_MODEL_ID)
    reply = conv.send(m.group(1).strip())
    (HERE / "transcripts").mkdir(exist_ok=True)
    (HERE / "transcripts" / "generator.json").write_text(json.dumps(conv.history, indent=2))
    data = _extract_json(reply)
    assert len(data["sender_vignettes"]) == 24 and len(data["counterpart_vignettes"]) == 8
    (HERE / "vignettes.json").write_text(json.dumps(data, indent=2))
    print("wrote vignettes.json - REVIEW IT (SN/AA/NG), paste into vignettes.md, then run --rate")


# ---------------------------------------------------------------- rating
def msg0(vmap, sender):
    lines = [f"V{i + 1}: {sender[vid]}" for i, vid in enumerate(vmap)]
    return (
        "I am running a research project on how people perceive the similarity between everyday "
        "situations. I will first give you 24 short vignettes, each describing a situation "
        "(labeled V1 to V24). In later messages I will show you other texts and ask you to rate "
        "their similarity to these vignettes. For now, read the vignettes and reply only: "
        '"Ready."\n\n' + "\n\n".join(lines)
    )


def msg_context(letter, text):
    return f"""Here is a text describing a situation.

TEXT {letter}:

{text}

TASK 1. For each vignette V1-V24, rate on a 0-100 scale how similar the situation in TEXT {letter} \
is to the situation in the vignette, considering the situation as a whole: what is at stake, the \
relationship between the parties, the setting, and the structure of the decision each party faces. \
0 = the two situations have nothing in common; 100 = they are essentially the same situation.

TASK 2. Now imagine a person in the situation of TEXT {letter}, asking themselves: "What kind of \
situation is this? What does it remind me of?" Distribute exactly 100 points across the 24 \
vignettes according to how strongly each would come to mind as a similar experience. You may give \
0 points to any vignette.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"ratings": {{"V1": <0-100>, ..., "V24": <0-100>}}, "retrieval_points": {{"V1": <int>, ..., "V24": <int>}}}}
The retrieval points must sum to exactly 100."""


def msg_receiver(letter, game, rmap, counter):
    lines = [f"R{i + 1}: {counter[rid]}" for i, rid in enumerate(rmap)]
    return f"""Consider again TEXT {letter}. In that situation there is another party on the other \
side: {RECEIVER_ROLE[game]}.

Here are four descriptions of possible counterparts:

{chr(10).join(lines)}

Distribute exactly 100 points across R1-R4 according to how strongly each description matches the \
counterpart you would picture on the other side of the situation in TEXT {letter}.

OUTPUT FORMAT: return a single JSON object, no other text:
{{"points": {{"R1": <int>, "R2": <int>, "R3": <int>, "R4": <int>}}}}
The points must sum to exactly 100."""


def run_rate(models, sets):
    import csv
    contexts = load_contexts()
    sender, counter = load_vignettes()
    (HERE / "transcripts").mkdir(exist_ok=True)
    ratings_f = HERE / "recording_round2.csv"
    splits_f = HERE / "recording_splits_round2.csv"
    new = not ratings_f.exists()
    with open(ratings_f, "a", newline="") as fr, open(splits_f, "a", newline="") as fs:
        wr, ws = csv.writer(fr), csv.writer(fs)
        if new:
            wr.writerow("model set context vignette_id label_context label_vignette similarity_rating".split())
            ws.writerow("model set task context item_id points".split())
        for model in models:
            for s in sets:
                print(f"== {model} set {s}")
                vmap = VIGNETTE_ORDER[s]
                conv = Conversation(model, MODEL_IDS[model])
                conv.send(msg0(vmap, sender))
                for letter in "ABCDEFGHIJ":
                    key = CONTEXT_LABELS[s][letter]
                    data = _extract_json(conv.send(msg_context(letter, contexts[key])))
                    for i, vid in enumerate(vmap):
                        wr.writerow([model, s, key, vid, letter, f"V{i + 1}",
                                     data["ratings"][f"V{i + 1}"]])
                        ws.writerow([model, s, "retrieval", key, vid,
                                     data["retrieval_points"][f"V{i + 1}"]])
                    print(f"   TEXT {letter} ({key}) done")
                for letter in "ABCDEFGHIJ":
                    key = CONTEXT_LABELS[s][letter]
                    if key in STRATEGIC:
                        game = STRATEGIC[key]
                        rmap = COUNTERPART_ORDER[(game, s)]
                        data = _extract_json(conv.send(msg_receiver(letter, game, rmap, counter)))
                        for i, rid in enumerate(rmap):
                            ws.writerow([model, s, "receiver", key, rid,
                                         data["points"][f"R{i + 1}"]])
                        print(f"   receiver module {key} done")
                (HERE / "transcripts" / f"rater_{model}_set{s}.json").write_text(
                    json.dumps(conv.history, indent=2))
                fr.flush(); fs.flush()
    print(f"wrote {ratings_f.name}, {splits_f.name}; transcripts in transcripts/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--rate", action="store_true")
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--sets", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    if not (args.generate or args.rate):
        ap.error("pass --generate and/or --rate")
    if args.generate:
        run_generate()
    if args.rate:
        models = _available(args.models)
        if not models:
            sys.exit("no usable models")
        run_rate(models, args.sets)
