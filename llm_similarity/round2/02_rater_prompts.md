# Rater prompts — round 2 (contexts × vignettes)

**Setup notes (May protocol).** Run each of the three conversation sets below in a separate, fresh
conversation with each model: 3 sets × 3 models = 9 conversations. Use the reasoning tier of each
model (as in May: Claude Opus 4.7, ChatGPT 5.5 Thinking, Gemini 3 Pro — update to current tiers if
re-run later). Within a conversation, send Message 0 first, wait for the full answer, then proceed
one message at a time, in order.

Labels (TEXT A–J for contexts, V1–V24 for sender vignettes, R1–R4 for counterpart descriptions)
refer to different underlying items in each set — deliberate, to test robustness to position and
labeling. The private mapping tables are at the end of this file; never include them in a prompt.

Vignette texts come from `vignettes.md` (the frozen generator output); context texts from
`contexts.md`. Where a message says `{V<k>: text}` or `{TEXT <letter>}`, paste the corresponding
text under that label per the set's mapping tables.

Record: the 24 ratings + the 100-point retrieval split from each context message, and the
100-point counterpart split from each receiver message, into `recording_template.csv` /
`recording_splits_template.csv` (one row per number; see headers there).

---

## Message 0 — materials

***[begin prompt]***

I am running a research project on how people perceive the similarity between everyday situations.
I will first give you 24 short vignettes, each describing a situation (labeled V1 to V24). In
later messages I will show you other texts and ask you to rate their similarity to these
vignettes. For now, read the vignettes and reply only: "Ready."

V1: {V1 text}
V2: {V2 text}
...
V24: {V24 text}

***[end prompt]***

## Messages 1–10 — one per context, in the set's TEXT order

***[begin prompt]***

Here is a text describing a situation.

TEXT {letter}:

{context text}

TASK 1. For each vignette V1–V24, rate on a 0–100 scale how similar the situation in TEXT {letter}
is to the situation in the vignette, considering the situation as a whole: what is at stake, the
relationship between the parties, the setting, and the structure of the decision each party faces.
0 = the two situations have nothing in common; 100 = they are essentially the same situation.
Output a table with columns Vignette | Rating. After the table, give a one-sentence justification
for your three highest and three lowest ratings only.

TASK 2. Now imagine a person in the situation of TEXT {letter}, asking themselves: "What kind of
situation is this? What does it remind me of?" Distribute exactly 100 points across the 24
vignettes according to how strongly each would come to mind as a similar experience. You may give
0 points to any vignette. Output a second table with columns Vignette | Points, and check that the
points sum to 100.

***[end prompt]***

## Messages 11–14 — counterpart module, only for the four strategic contexts

Send one such message for each of C-UG, M-UG, C-TG, M-TG (i.e., for the TEXT letters they carry in
this set), after Messages 1–10. Use the UG counterpart set (per mapping) for the two UG contexts
and the TG counterpart set for the two TG contexts.

***[begin prompt]***

Consider again TEXT {letter}. In that situation there is another party on the other side:
{for UG contexts: "the person who sees the proposed terms and decides whether to accept or reject
them" / for TG contexts: "the person who receives what is sent and decides how much to give
back"}.

Here are four descriptions of possible counterparts:

R1: {R1 text}
R2: {R2 text}
R3: {R3 text}
R4: {R4 text}

Distribute exactly 100 points across R1–R4 according to how strongly each description matches the
counterpart you would picture on the other side of the situation in TEXT {letter}. Output a table
with columns Counterpart | Points, and check that the points sum to 100.

***[end prompt]***

---

## PRIVATE mapping tables — do not include in any prompt

### Context labels

| TEXT | Set 1 | Set 2 | Set 3 |
|---|---|---|---|
| A | C-UG  | M-TG | AID  |
| B | BONUS | C-LT | M-LT |
| C | M-KW  | AID  | C-TG |
| D | C-TG  | M-UG | C-KW |
| E | AID   | C-KW | M-TG |
| F | M-UG  | BONUS| C-UG |
| G | C-KW  | M-LT | BONUS|
| H | M-TG  | C-UG | M-KW |
| I | C-LT  | C-TG | C-LT |
| J | M-LT  | M-KW | M-UG |

### Sender vignette labels (V1–V24)

Set 1: S3, C7, M1, C2, S8, M6, C4, S1, M8, C1, S5, M3, C8, S2, M7, C5, S6, M2, C3, S7, M4, C6, S4, M5
Set 2: M4, S6, C1, M2, C8, S3, M7, C5, S1, M5, C3, S8, M1, C6, S4, M8, C2, S7, M3, C7, S2, M6, C4, S5
Set 3: C6, M5, S2, C4, M8, S7, C1, M3, S5, C8, M1, S4, C3, M6, S8, C7, M2, S6, C5, M7, S3, C2, M4, S1

(read: in Set 1, V1 = generator vignette S3, V2 = C7, ..., V24 = M5)

### Counterpart labels (R1–R4)

| Module | Set 1 | Set 2 | Set 3 |
|---|---|---|---|
| UG | UR2, UA1, UR1, UA2 | UA2, UR1, UA1, UR2 | UR1, UA2, UR2, UA1 |
| TG | TK1, TR2, TR1, TK2 | TR1, TK2, TK1, TR2 | TK2, TR1, TR2, TK1 |

(read: in Set 1's UG module, R1 = UR2, R2 = UA1, R3 = UR1, R4 = UA2)
