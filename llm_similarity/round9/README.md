# Round 9: DG-C wording sensitivity and model-run intervals

This folder preserves round 8 and repeats its two-stage relative-similarity
exercise after a minimal revision of the two DG-C vignettes. The phrases
describing the sender as happy/pleased and emphasizing the sender's remaining
payoff were removed. The complementary-contribution, joint-value, and sharing
logic was retained.

`00_neutralize_dg_c_wording.py` makes exactly two paragraph replacements in
the round-8 DOCX and writes
`input/joint_action_vignettes_dg_c_neutral.docx`. The exact before/after text
is recorded in `dg_c_wording_changes.json`. Structural comparison confirms
that no other paragraph changed. Both revised narratives remain classified as
Cooperation (prompt output 2) under the round-7 HP classification prompt; the
check is recorded in `dg_c_revision_classification_check.csv`.

The remaining workflow mirrors round 8:

1. `01_prepare_revised_inputs.py` validates and anonymizes 22 vignettes and 12
   game-frame contexts.
2. `02_build_blind_relative_packets.py` creates three separately shuffled,
   classification-free packets.
3. Three agents independently allocate 1,000 integer points within every task.
4. `03_merge_blind_relative_ratings.py` validates and merges all ratings.
5. `04_analyze_relative_similarity.py` computes point estimates and 95 percent
   t-intervals across the three agent-level estimates, then writes the tables
   and figures used by `additional_similarity.tex`.

The confidence intervals measure variability across the three model runs. They
are not participant-sampling intervals and are necessarily imprecise with
only three runs.

## Completion status

The workflow is complete. Each rater file contains 352 validated rows and all
24 allocations per rater sum to exactly 1,000 points. The merged file contains
1,056 raw ratings and 352 three-rater candidate means. Agent-level estimates,
paired frame differences, t-intervals, and the four final figures are stored
in `relative_output/`.

For DG, the Market-minus-Control Cooperation shift is 1.7 percentage points
(95 percent model-run interval: 0.2 to 3.1), compared with 3.9 points before
the wording revision. The Moral shift is -2.8 points (-4.7 to -0.9).

## Manuscript hybrid output

The full round-9 rerating is retained for reproducibility, but it is not used
for UG or TG in the manuscript. `05_build_hybrid_outputs.py` creates
`hybrid_output/` with the intended specification:

- DG uses the round-9 ratings based on the neutralized DG-C texts.
- UG and TG use the original round-8 ratings.
- Confidence intervals for each game use the three agents from that game's
  source round.

`additional_similarity.tex` points to `hybrid_output/`.
