# Final vignette-similarity workflow

This folder contains the exact vignette materials behind Sections 1--2 of
`../../../additional_similarity.tex`.

## Final vignette set

- `input/joint_action_vignettes_final.docx`: authoritative 22-vignette source.
- `input/vignettes_catalog.csv`: machine-readable extraction with construction
  metadata.
- `input/game_contexts.md`: source game/frame instructions used as targets.
- `documentation/vignette_design.md`: construction scheme and interpretation.
- `documentation/dg_c_wording_changes.json`: exact before/after record for the
  final neutralization of the two DG-Cooperation narratives.
- `documentation/dg_c_revision_classification_check.csv`: confirmation that both
  edited narratives remain Cooperation under the HP classification prompt stored
  at `../hp_similarity/prompts/classification_prompt.txt`.

## Rating protocol

The exact agent instructions are in `prompts/relative_similarity_prompt.md`.
Each of three agents saw only its independently shuffled neutral packet. For every
task it allocated exactly 1,000 nonnegative integer similarity points jointly
across either all 22 vignettes (Stage 1) or the vignettes from the target game
(Stage 2). Agents did not see vignette classifications, source identifiers, other
ratings, analysis code, or results.

The stored protocol identifies the raters as three separately tasked Codex agents
from the same model family; an exact model-version string was not recorded in the
rating files. The exact prompt and complete blinded packets are retained.

The manuscript specification is deliberately retained:

- DG uses the rating run after the two DG-C texts were neutralized;
- UG and TG use the original revised-vignette rating run;
- the UG/TG texts are identical in the final DOCX, so only their original ratings
  are retained, not an obsolete alternative vignette set.

The two complete audit trails are under `rating_runs/original/` and
`rating_runs/dg_c_neutral/`. The former includes the exact pre-edit DOCX because it
is the source used for the retained UG/TG ratings.

## Reproduction sequence

Run from this directory:

```text
python code/00_rebuild_final_vignettes.py
python code/01_prepare_blinded_inputs.py
python code/02_build_blinded_packets.py
```

Three blinded agents then use `prompts/relative_similarity_prompt.md` and their
corresponding file in `rating_runs/dg_c_neutral/relative_packets/`, writing the
three `relative_agent*_ratings.csv` files to that run directory. Continue with:

```text
python code/03_merge_ratings.py
python code/04_analyze_similarity.py
python code/05_build_manuscript_outputs.py
```

`04` records the full current-run analysis as a sensitivity output.
`05` writes the manuscript specification to `output/`. Confidence intervals are
two-sided Student-t intervals across the three agent-level estimates (df=2); they
measure model-run variability, not participant-sampling uncertainty.

Python dependencies are listed in `requirements.txt`.

