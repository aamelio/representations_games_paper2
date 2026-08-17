# Round-1 rerun — neutral similarity prompt (NG meeting 2026-07-23, item 3)

**Status: DRAFT for AA's sign-off** (SN + Claude, 2026-07-28). Based on AA's proposal
(email 2026-07-24), with three amendments agreed with SN: (i) the label-permutation
protocol of the May run is restored (three triplets, private mappings — AA's draft had a
single fixed ordering); (ii) the output format keeps the May order (per-pair rating and
justification first, summary table at the end) so ratings are not committed before the
reasoning; (iii) the clause "even if their contextual details differ" is trimmed from the
100-anchor — it is a softened residue of the instruction the rerun is meant to remove.

**What changes vs. May:** only Prompt 1 below (it replaces Prompt 1.1/2.1/3.1 of
`memory_games_llm_prompts.docx`). It drops the "Focus on the structural and strategic
features … not on surface thematic content" sentence and the five-feature checklist, and
instead leaves the weighting of aspects to the rater. Prompts 2 (structural
decomposition) and 3 (retrieval split) run verbatim as in May, after Prompt 1, in the
same conversation. Same three triplets and label mappings as the May doc; same models
(reasoning tier of each provider, updated to current versions); one fresh conversation
per model x triplet.

**Purpose:** robustness of Table 5's two consumed findings (the DG-KW > DG-LT > UG > TG
structural-distance ordering; Bonus ≈ Aid) to the guided instruction. Placement decision
(replace Table 5 vs. robustness appendix) after the numbers are in.

---

## Prompt 1 (amended) — send first in each conversation

***[begin prompt — copy everything between the two rules]***

I am running a research project comparing real-world stories with decision situations. I
will give you descriptions of four games and two short stories.

INSTRUCTIONS

Below are four games (A, B, C, D) and two stories (Story 1, Story 2).

For each of the 8 story x game pairs, rate how similar the two situations are in their
underlying decision structure on a 0-100 scale:

0 = the situations have no meaningful similarity in their underlying decision structure

100 = the situations have essentially the same underlying decision structure

Assess each pair independently and base your judgment only on the descriptions provided.
Use your own judgment to decide which aspects of the situations are relevant and how much
weight to give them. Apply the same standard across all eight pairs. Equal ratings are
allowed, and the ratings do not need to sum to any particular total.

OUTPUT FORMAT

1. For each of the 8 pairs, give the rating and a 2-3 sentence justification identifying
the main considerations behind the rating.

2. At the end, produce a single summary table with games as rows (A, B, C, D) and stories
as columns (Story 1, Story 2).

3. Below the table, briefly identify the most and least similar game or games for each
story and explain why. Allow for ties.

[game and story texts follow, in the triplet's label order — identical materials to the
May doc]

***[end prompt]***

---

## Protocol notes

- Triplet label mappings: use the May doc's three private mapping tables unchanged
  (Triplet 1: A=DG-KW, B=DG-LT, C=UG, D=TG, Story 1=Bonus, Story 2=Aid; Triplets 2-3 as
  recorded there). Never include the mappings in a prompt.
- Prompt 2 and Prompt 3 of the May doc: unchanged, sent after Prompt 1 in the same
  conversation, in that order. They cannot contaminate Prompt 1's ratings (sent after)
  and keep the decomposition and retrieval-split columns comparable across runs.
- Recording: same workbook layout as `memory_games_llm_recording.xlsx`, new sheet or
  copy tagged `rerun_neutral`.
- Execution: manual chats as in May, or via API where keys are available (the round-2
  runner's provider adapters can be reused; GPT/Gemini ids in
  `round2/03_api_runner.py::MODEL_IDS` still need updating).
