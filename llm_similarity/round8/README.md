# Round 8: revised-vignette relative similarity

This folder repeats the round-6 two-stage, 1,000-point relative-similarity
exercise using the revised vignette set supplied in
`input/joint_action_vignettes_revised.docx`. Round 6 and the round-7 HP
analysis remain unchanged.

The workflow is deliberately checkpointed through files:

1. `01_prepare_revised_inputs.py` extracts and validates 22 vignettes (6 DG,
   8 UG, 8 TG), reconstructs the 12 existing game-frame contexts, and writes
   neutral inputs plus a private mapping.
2. `02_build_blind_relative_packets.py` creates one separately shuffled
   classification-free packet for each of three independent raters.
3. Each rater uses only `relative_rating_prompt.md` and its assigned packet,
   then writes `relative_agentN_ratings.csv`.
4. `03_merge_blind_relative_ratings.py` validates candidate coverage, integer
   allocations, and the 1,000-point constraint before combining the passes.
5. `04_analyze_relative_similarity.py` writes tables and figures to
   `relative_output/`.

Stage 1 uses all 22 vignettes. Stage 2 uses only same-game vignettes. The
classifications are joined only after the blinded ratings have been recorded.

## Completion status

The workflow is complete. The three validated rater files contain 352 rows
each (24 tasks per rater); `relative_ratings_3agents.csv` contains 1,056 raw
ratings and `relative_ratings_3agent_means.csv` contains 352 three-rater means.
Every task allocates exactly 1,000 nonnegative integer points. The four final
figures and their underlying CSV tables are in `relative_output/`.
