# Round-3 similarity module — design decision (NG meeting 2026-07-23, item 5)

**Decided 2026-07-28 (SN + Claude); to be confirmed with AA and NG.** AA's reading of the
meeting's "6 categories" (3 attention categories x High/Low sender belief) is adopted for
the sender side; the receiver side is covered not by a parallel 3-trait receiver pool but
by new receiver vignettes keyed to the four categories Player-2 answers are actually
classified into. Rationale: (i) a 3-trait receiver pool would match no downstream exhibit
(the P2 scheme has four categories); (ii) P2-keyed receiver vignettes allow a
receiver-side analogue of Figure 6 (receiver similarity shifts vs. measured P2
representation shifts), a new exhibit; (iii) NG's "morality, selfishness, cooperativeness
of sender + receiver" is then covered in full, with the receiver margin in the units the
paper already uses.

## The pool

**Sender side (48):** AA's draft `../vignettes_6rep_draft.md` — 3 attention categories
(Moral, Self-interest, Mutual Benefit/Cooperation) x sender belief about the counterpart
(High/Low) x 8 vignettes (4 individuals, 4 firms/market). Review points to settle with
AA before freezing (from SN + Claude review, 2026-07-28; none blocking):

1. *Moral loses the passive recipient.* Giving the sender a belief requires counterpart
   agency, so the new Moral vignettes drift from the round-2 CLASS M rule ("the other
   person cannot affect the outcome") toward trust-game-like structure; DG contexts may
   mechanically rate less Moral-similar in round 3. Accept and note, or keep the
   category margin (average over belief levels) as the DG-relevant comparison.
2. *Self-interest x Low belief entanglement.* The protective prong makes the low belief
   nearly the reason for the category; AA's texts mostly separate the two, but this is
   the cell to watch in the ratings.
3. Mechanical checks passed: no banned words in any vignette body; 48 distinct
   protagonist names; individuals/firms balanced 4/4 in every cell.

**Receiver side (16, replacing the old 8 binary accept/reject-return/keep vignettes):**
four vignettes per P2 class — Moral good, Moral bad, Self-interest, Mutual
Benefit/Cooperation — each class balanced 2 individuals / 2 firms and 2
respond-to-proposed-terms / 2 entrusted-proceeds settings, so class is confounded neither
with the character type nor with the game-shaped setting. Generator prompt in
`01_receiver_generator_prompt.md`; draft output in `receiver_vignettes_draft.md` for
SN/AA/NG review (same freeze convention as round 2). One review point from the first
read (SN + Claude, 2026-07-28): in two of the four indignant-class drafts (B2, B4) the
condemned conduct targets a third party rather than the responder, whereas the games'
Moral-bad receivers condemn conduct aimed at themselves (the low offer they face);
consider regenerating those two with responder-directed grievances.

## Scoring

- Contexts x sender cells: graded 0-100 (as Table 6). Category similarity = mean over the
  cell's 8 vignettes and both belief levels; belief similarity = High minus Low within
  category — the similarity counterpart of the representation's belief component, and the
  instrument for the flagged TG-Market anomaly (representations and beliefs move, category
  similarity does not).
- Strategic contexts x receiver classes: graded 0-100 (as Table 41), plus retrieval
  splits for continuity with Table 40.
- Rater protocol: adapt `../round2/02_rater_prompts.md` (labels V1-V48, R1-R16, permuted
  sets, blind ex-post mapping). Details after the pool is frozen.
