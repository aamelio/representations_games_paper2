# Final HP-similarity workflow

This folder contains the exact inputs and audit trail behind Section 3 of
`../../../additional_similarity.tex`.

## Design

The source contains 4,800 hypothetical-allocation descriptions: four allocation
anchors for each of 1,200 participant-game-condition frames. The 1,200 HPmin labels
are inherited by exact match; the remaining 3,600 descriptions were classified by
GPT-5.6 using `prompts/classification_prompt.txt`.

Each description was then rated against the DG-KW Control instructions and,
separately, the DG-KW Market instructions. Three independently assigned blinded
agents used `prompts/similarity_prompt.md` and supplied an integer 0--100 score for
each description-reference pair. The reported row-level score is their arithmetic
mean.

The packets expose only a neutral response identifier and text. Similarity raters
do not see source condition, category, allocation anchor, or the substantive name
of the reference text.

## Files

- `input/`: raw HP source workbooks and the two frozen reference instructions.
- `prompts/`: exact classification and similarity prompts.
- `work/`: blinded packets, completed agent outputs, and resumable checkpoints.
- `data/hp_responses_classified_and_rated.*`: final 4,800-row dataset.
- `output/`: validation manifests, rater agreement, plot values, and manuscript
  figures.

## Reproduction sequence

Run from this directory:

```text
python code/01_prepare_inputs.py
python code/02_resume_and_finalize.py progress
python code/02_resume_and_finalize.py finalize
python code/03_plot_similarity.py
```

If a new rating run is interrupted, `progress` identifies the next incomplete
packet. Existing completed packet outputs are retained under `work/` so the final
dataset can also be audited without rerunning the agents.

Python dependencies are listed in `requirements.txt`.

