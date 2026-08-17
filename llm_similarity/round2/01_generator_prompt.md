# Generator prompt — vignette set (run once, freeze output in `vignettes.md`)

Run in a single fresh conversation with the designated generator model. Record model + date at the
top of `vignettes.md`. Review the output (SN/AA/NG) before any rating conversation starts; after
rating starts, the set is frozen.

> **Reconciled 2026-07-19** against the classification-rules file (`instruments/prompts_pl1_pl2.docx`):
> CLASS S extended with the strategic self-protection prong of P1 category 3 (distrust /
> betrayal-avoidance — half the S vignettes use it); CLASS C given category 2's risk-with-upside
> clause; "greedy" and "betray" added to the banned words. CLASS M matched category 1 as written.

---

***[begin prompt — copy everything between the two rules]***

I am preparing materials for a research project on how people categorize everyday social and
economic situations. I need you to write short vignettes — descriptions of realistic everyday
situations — that exemplify some general classes of situations. Please follow the specification
exactly.

PART 1 — SENDER-SIDE VIGNETTES (24 in total)

Write 8 vignettes for each of the following three classes of situations:

CLASS M (moral/sharing problem): a situation in which one person controls a resource and must
decide how much of it to share with another person, and the decision is governed by considerations
of fairness, generosity, or equality — what it would be right to do, what the other person deserves
or needs, whether an even division is called for. The other person cannot affect the outcome.

CLASS S (own-payoff problem): a situation in which one person controls a resource and decides how
much of it to keep, and the decision is governed by their own material interest — what they get to
keep, what they need it for, what it cost them to obtain — or by protecting what is theirs against
the risk that the other person would not do their part, give nothing back, or take advantage of
them. The other person's benefit is not what drives the decision. Four of the 8 vignettes should
use the protective logic (guarding against the other side), four the pure own-use logic.

CLASS C (joint-gains problem): a situation in which two people can both end up better off if one of
them entrusts, contributes, or hands over resources so that the pair produces or secures more
together than either could alone — teamwork, exchange, or investment where the point of the
interaction is enlarging the total, and where the outcome depends on both sides doing their part.
The situation may involve some risk that the other side does not follow through, as long as what
drives the decision is the prospect of the larger joint gain.

Requirements for every vignette:
1. 60–90 words, third person, one named protagonist and exactly one named or clearly identified
   counterpart (a person, not an organization).
2. A concrete resource is at stake (money, goods, time, produce, equipment — vary it), and the
   protagonist faces a concrete decision about it.
3. Do NOT use the words "fair", "unfair", "selfish", "generous", "greedy", "cooperate",
   "cooperation", "moral", "trust", "betray", or their derivatives: the class must be conveyed by
   what the situation is, not by naming it.
4. Do not mention experiments, studies, games, players, laboratories, or questionnaires.
5. Do not use the dollar amounts 6, 12, or 18, and do not use a tripling of any amount.
6. Domains: within each class, assign the 8 vignettes to these domains — one each to (i) workplace,
   (ii) family or friends, (iii) neighbors or local community, (iv) strangers in a one-off
   encounter, (v) small business or commerce, (vi) online or anonymous interaction, and two free
   choices. Every class must use the same six fixed domains so that class and setting are not
   confounded.
7. Vary protagonist names, genders, and ages across vignettes; do not reuse a name.

PART 2 — COUNTERPART VIGNETTES (8 in total)

Now write 8 short descriptions (40–60 words each, third person, one named person) of a
*counterpart* — someone on the receiving end of a two-party dealing:

- 2 of type UA: a counterpart who, when offered terms by the other side, goes along with any terms
  that leave them better off than nothing, however one-sided.
- 2 of type UR: a counterpart who turns down one-sided terms even at a cost to themselves, out of
  indignation at how they are being treated.
- 2 of type TR: a counterpart who, having been handed control over resources the other side
  provided, gives back a substantial share of the proceeds.
- 2 of type TK: a counterpart who, having been handed control over resources the other side
  provided, keeps the proceeds for themselves.

The same requirements 3–5 and 7 apply. Spread the 8 descriptions across different domains.

OUTPUT FORMAT

Return a single JSON code block:

```json
{
  "sender_vignettes": [
    {"id": "M1", "class": "M", "domain": "workplace", "text": "..."},
    ... 24 items: M1-M8, S1-S8, C1-C8 ...
  ],
  "counterpart_vignettes": [
    {"id": "UA1", "type": "UA", "text": "..."},
    ... 8 items: UA1, UA2, UR1, UR2, TR1, TR2, TK1, TK2 ...
  ]
}
```

***[end prompt]***
