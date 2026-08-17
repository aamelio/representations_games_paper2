"""New-scheme classification of the hp-allocation memory texts (NG p.22, item P6),
plus the cross-model validation of the classifier swap (GPT-4.1 -> Claude Opus 4.8).

The hypothetical-allocation answers were never classified with the new category scheme
(AA 2026-07-10: OpenAI credit issues); the original Reasons classifier model may no longer
be available. This script therefore (a) validates a Claude Opus 4.8 classifier against the
existing GPT-4.1 labels on a stratified sample of Reasons answers, with the written
classification prompt VERBATIM (instruments/prompts_pl1_pl2.docx, Player 1 section), and
(b) classifies all 1,200 hp memory texts (hpmin_social_proximity_all.xlsx) with the same
category definitions; only the two preamble paragraphs are adapted, because the hp texts
describe real-world situations rather than justify game decisions. Benchmark for (a):
human-coder vs GPT-4.1 agreement on the P1 batches was 79-80% (kappa .63-.69).

Usage (needs ANTHROPIC_API_KEY):
  python3 14_hp_classification.py --submit    # submit both batch jobs, save batch ids
  python3 14_hp_classification.py --collect   # poll until ended, write outputs

Outputs (output/hp_classification/):
  batch_state.json, results_validation.csv, results_hp.csv, validation_stats.txt
Data deliverable: data/hpmin_new_scheme_categorized.xlsx (hp file + category / category_num).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "output" / "hp_classification"
MODEL = "claude-opus-4-8"
SEED = 20260719
PER_CELL = 50  # validation draws per game x category cell (4 x 4)
MAX_TOKENS = 16  # "Output only the number" -- strict parse below

CATEGORY_NAMES = {0: "No clear justification", 1: "Moral",
                  2: "Mutual Benefit / Cooperation", 3: "Self-interest"}

# --- Player 1 classification prompt, verbatim from instruments/prompts_pl1_pl2.docx ----
P1_PREAMBLE = """\
You are a helpful assistant classifying free-text survey responses based on the type of \
reasoning used to justify decisions in strategic interaction games, where Player 1 (the \
sender / proposer / decision-maker) chooses how much to send, offer, or allocate to \
another player.

The response may come from the Dictator Game (DG), Trust Game (TG), or Ultimatum Game (UG)."""

# Adapted preamble for the hp memory texts (the ONLY divergence from the verbatim prompt;
# category definitions, guidelines, and tie-breakers below are shared verbatim).
HP_PREAMBLE = """\
You are a helpful assistant classifying free-text survey responses. Each response \
describes a real-world situation the respondent finds similar to a hypothetical \
allocation of money between themselves and another person. Classify the situation \
described based on the type of reasoning or interpersonal logic it depicts."""

P1_BODY = """\
0 = No clear justification

- No meaningful explanation is given, or the response is too vague.
- Merely restates the action without explaining why.
- Example: "I sent that amount because it seemed fine."

1 = Moral

- Appeals to fairness, kindness, sharing, generosity, or doing what is "right."
- Focuses on moral or social norms rather than on monetary returns from cooperation, \
joint value creation, or strategic protection of one's own payoff.
- Includes fairness, equality, equal sharing, or balanced final outcomes.
- Includes wanting both players to end up with the same or a fair amount, when the main \
logic is fairness rather than surplus creation.
- Typical language: "fair," "right," "deserve a fair share," "equal," "half," "50-50 \
split," "share," "help," "generous," "the right thing," "same amount," "even split," \
"balanced," "fair outcome."
- Examples:
- "I wanted to be fair."
- "It felt like the right thing to do."
- "I didn't want to be greedy."
- "I wanted to share some of it."
- "We will both end up with the same amount."
- "I chose this because it seemed like the fairest outcome for both of us."

2 = Mutual benefit / productive partnership reasoning

- Frames the decision as a productive or value-creating opportunity due to the \
cooperation between the two parties. It instead emphasizes less or not at all the moral \
fairness or justice of the payoff distribution.
- Emphasizes the higher overall profits obtained by the two players, the joint returns, \
profitability, value creation, positive-sum reasoning, cooperation, or both players \
benefiting.
- In the Trust Game, this includes classic investment logic.
- May explicitly mention risk, uncertainty, hedging, or a "safety net" to justify the \
amount transferred, but the main focus remains on the potential for larger joint surplus \
or a better overall outcome.
- This category should also include cases where the participant mentions their own gain, \
as long as the reasoning is still mainly about creating more value through exchange or \
achieving an overall larger amount of resources for both players.
- Typical language: "lucrative," "most profitable," "increase my return," "expected \
return," "upside/downside," "hedge," "safety net," "risk but worth it," "both benefit," \
"better overall," "we both gain."
- Examples:
- "It's a good investment since the amount increases."
- "We increase profits if I send more."
- "I wanted a chance to increase returns but kept some as a safety net."
- "It's risky, but potentially more lucrative if we both cooperate."
- "This way there is more overall."
- "I wanted something that could benefit both of us."

3 = Strategic self-protection / self-interest reasoning

- Focuses on protecting, securing, or maximizing one's own outcome as the primary \
justification.
- More generally, this category includes reasoning centered on keeping more for oneself, \
prioritizing one's own material payoff, avoiding personal losses at the possible expense \
of a social gain, minimizing downside risk stemming from the other player's refusal to \
cooperate, or prioritizing one's own payoff without a mainly productive or mutually \
beneficial logic.
- It emphasizes a self-regarding or protective logic: "If they were me, they would not \
give anything", or "I don't trust them / they may keep it / I might get nothing back / I \
wanted to keep more for myself," rather than "it's a risky investment with upside for \
both."
- Typical language: "they would not give", "I don't trust them," "they'll keep it," \
"they might take advantage," "I'd rather be safe than sorry," "I wanted to keep more," \
"better for me," "protect my payoff," "something is better than nothing."
- Examples:
- "They might keep everything, so I sent little."
- "I don't trust the other player to return anything."
- "I was afraid they would take advantage of me."
- "I avoided sending more because I could be betrayed and get nothing back."
- "I wanted to keep as much as possible."
- "I chose the amount that was best for me."

Guidelines:

- Classify based only on what is explicitly stated in the response.
- Do not infer motives that are not mentioned.
- The model MUST pick exactly one category (0-3). There is no mixed option.

Key distinction between 1 and 2:

- Choose 1 when the response is mainly about fairness, equality, generosity, decency, or \
both players ending up with a fair or equal outcome.
- Choose 2 when the response is mainly about investment, returns, value creation, \
profitability, multiplier effects, or making more overall.
- If "both benefit" means "both end up with a fair or equal amount," choose 1 because in \
this case mentioning both players does not entail higher social returns.
- If "both benefit" means "the interaction creates more social value or higher returns," \
choose 2.

Key distinction between 2 and 3:

- Choose 2 when the response is framed as an INVESTMENT / UPSIDE / MUTUAL BENEFIT \
decision, even if it mentions risk, hedging, uncertainty, or keeping some guaranteed \
amount.
- Choose 3 only when self-protection, betrayal-avoidance, or one's own payoff is the \
MAIN reason for sending less, offering less, or keeping more.

If a response contains multiple elements:

- Choose the category emphasized as the main reason (e.g., introduced by "because," "so \
that," "I wanted to").
- If emphasis is unclear, use these tie-breakers:

* mentions profit / return / upside / value creation / multiplier / both benefit / \
better overall / hedging / safety net -> 2
* mentions betrayal / exploitation / don't trust / they'll keep it / I might get \
nothing / protect my payoff / keep more for myself as the main rationale -> 3
* purely moral / fairness / generosity language -> 1

- Use 0 only when no clear justification is present.

Output only the number (0-3)."""

P1_SYSTEM = f"{P1_PREAMBLE}\n\n{P1_BODY}"
HP_SYSTEM = f"{HP_PREAMBLE}\n\n{P1_BODY}"


def load_validation_sample() -> pd.DataFrame:
    p1 = pd.read_excel(DATA / "player1_all_categorized.xlsx")
    p1 = p1[p1.reasons.notna()].reset_index(drop=True)
    parts = [g.sample(n=min(PER_CELL, len(g)), random_state=SEED)
             for _, g in p1.groupby(["game", "category"])]
    return pd.concat(parts).reset_index(drop=True)


def load_hp() -> pd.DataFrame:
    hp = pd.read_excel(DATA / "hpmin_social_proximity_all.xlsx")
    hp["game"] = hp.treatment.map({"kw": "dgkw", "lt": "dglt"})
    return hp.reset_index(drop=True)


def build_requests(texts: list[str], system: str, prefix: str) -> list[dict]:
    return [{
        "custom_id": f"{prefix}-{i}",
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": str(t)}],
        },
    } for i, t in enumerate(texts)]


def submit() -> None:
    import anthropic
    client = anthropic.Anthropic()
    OUT.mkdir(parents=True, exist_ok=True)

    val = load_validation_sample()
    hp = load_hp()
    val.to_csv(OUT / "validation_sample.csv", index=False)

    state = {}
    for name, df, system, col in [("validation", val, P1_SYSTEM, "reasons"),
                                  ("hp", hp, HP_SYSTEM, "memory")]:
        reqs = build_requests(df[col].tolist(), system, name)
        batch = client.messages.batches.create(requests=reqs)
        state[name] = {"batch_id": batch.id, "n": len(reqs)}
        print(f"submitted {name}: {batch.id} ({len(reqs)} requests)")
    (OUT / "batch_state.json").write_text(json.dumps(state, indent=2))


def _parse_label(text: str) -> int | None:
    m = re.fullmatch(r"[0-3]", text.strip())
    return int(m.group()) if m else None


def collect() -> None:
    import time

    import anthropic
    import numpy as np
    client = anthropic.Anthropic()
    state = json.loads((OUT / "batch_state.json").read_text())

    for name, info in state.items():
        while True:
            b = client.messages.batches.retrieve(info["batch_id"])
            if b.processing_status == "ended":
                break
            print(f"{name}: {b.processing_status} "
                  f"({b.request_counts.processing} processing)", flush=True)
            time.sleep(60)
        labels, failures = {}, []
        for r in client.messages.batches.results(info["batch_id"]):
            idx = int(r.custom_id.split("-")[1])
            if r.result.type != "succeeded":
                failures.append((idx, r.result.type))
                continue
            text = next((blk.text for blk in r.result.message.content
                         if blk.type == "text"), "")
            lab = _parse_label(text)
            if lab is None:
                failures.append((idx, f"unparseable: {text!r}"))
            else:
                labels[idx] = lab
        if failures:
            print(f"{name}: {len(failures)} failures/unparseable: {failures[:10]}")
        pd.Series(labels, name="claude_category_num").rename_axis("row").to_csv(
            OUT / f"results_{name}.csv")
        print(f"{name}: {len(labels)}/{info['n']} labels written")

    # -- validation stats: Claude vs GPT-4.1 on the Reasons sample -----------------------
    val = pd.read_csv(OUT / "validation_sample.csv")
    res = pd.read_csv(OUT / "results_validation.csv", index_col="row")
    val = val.join(res, how="inner")
    a, b = val.category_num.to_numpy(), val.claude_category_num.to_numpy()
    agree = (a == b).mean()
    K = 4
    po = agree
    pe = sum((a == k).mean() * (b == k).mean() for k in range(K))
    kappa = (po - pe) / (1 - pe)
    lines = [
        f"Classifier-swap validation: Claude Opus 4.8 vs original GPT-4.1 labels, "
        f"P1 prompt verbatim, stratified sample (seed {SEED}, up to {PER_CELL} per "
        f"game x category cell), N={len(val)}.",
        f"Agreement {agree:.3f}; Cohen's kappa {kappa:.3f} "
        f"(human-coder benchmark on the P1 batches: 79-80% agreement, kappa .63-.69).",
        "", "Confusion (rows = GPT-4.1, cols = Claude):",
        pd.crosstab(val.category_num, val.claude_category_num).to_string(),
        "", "Agreement by game:",
        val.assign(ok=a == b).groupby("game").ok.mean().round(3).to_string(),
        "", "Agreement by category (GPT-4.1 label):",
        val.assign(ok=a == b).groupby("category").ok.mean().round(3).to_string(),
    ]
    (OUT / "validation_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))

    # -- deliverable: hp file + new-scheme labels ----------------------------------------
    hp = load_hp()
    hp_res = pd.read_csv(OUT / "results_hp.csv", index_col="row")
    hp = hp.join(hp_res, how="left")
    hp["category_num"] = hp.claude_category_num
    hp["category"] = hp.category_num.map(CATEGORY_NAMES)
    hp.drop(columns=["claude_category_num"]).to_excel(
        DATA / "hpmin_new_scheme_categorized.xlsx", index=False)
    print(f"wrote data/hpmin_new_scheme_categorized.xlsx "
          f"({hp.category_num.notna().sum()}/{len(hp)} classified)")
    print(hp.category.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if not (args.submit or args.collect):
        ap.error("pass --submit and/or --collect")
    if args.submit:
        submit()
    if args.collect:
        collect()
