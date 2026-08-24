# Round 7: hypothetical-allocation memories

This folder contains the reproducible inputs and outputs for two row-level LLM tasks:

1. classify every hypothetical-allocation (HP) description into the paper's Player-1
   categories; and
2. rate every HP description separately against the DG-KW Control and Market
   instruction texts on an absolute 0-100 similarity scale, using three blinded
   rating passes and taking their row-level arithmetic mean.

The 1,200 HP responses previously retained as HPmin preserve their existing labels from
`replication_package/data/hpmin_new_scheme_categorized.xlsx`. Only the remaining 3,600
responses receive new GPT-5.6 labels. Matching uses the unique combination of participant
ID, DG treatment, HP allocation anchor, and exact memory text.

The blinded packets contain only a neutral HP-response identifier and the response text.
Similarity raters do not see treatment of origin, category, HP allocation anchor, or the
substantive identity of the reference text.

Files ending in `_used` record the exact prompts. `private_reference_map.json` maps the
neutral references back to the experimental contexts after ratings are complete.

The final dataset preserves identifiers, raw text, allocation anchor, source frame,
category and its provenance, all three raw ratings per reference, and the two mean
similarity ratings.

## Resumable workflow and final outputs

`02_resume_and_finalize_round7.py` validates progress from saved packet files. A packet
counts as complete only if it has the expected identifiers in the expected order and a
valid integer output for every row. Run:

```text
python 02_resume_and_finalize_round7.py progress
```

to recover the exact next missing packet after an interruption. The current state is also
written to `round7_progress.json`. The last 400 classifications were checkpointed in eight
50-row files under `classification_checkpoints/`; similarity outputs are checkpointed in
400-row files under `similarity_rater_outputs/`.

The completed outputs are:

- `hp_round7_final.csv`
- `hp_round7_final.xlsx`
- `similarity_rater_agreement.csv`
- `round7_final_manifest.json`

The final dataset has 4,800 rows: 1,200 categories inherited from HPmin and 3,600 new
GPT-5.6 classifications. Each HP response has three independent 0-100 ratings against
each neutral reference; the substantive reference identities are reattached only in the
final dataset, and the reported mean columns average the three ratings arithmetically.

## HP similarity figures

`03_plot_hp_similarity.py` reads `hp_round7_final.csv` and writes the figures used in
`../../additional_similarity.tex` to `figures/`:

- `hp_similarity_by_source_condition.png`
- `hp_similarity_by_category_within_condition.png`
- `hp_similarity_plot_values.csv`

The figures pool DG-KW and DG-LT. Bar heights are response-level means of the three-rater
averages. The 95% confidence intervals use participant-clustered standard errors because
each participant contributes four hypothetical-allocation responses. The category figure
excludes `No clear justification` and orders the substantive categories as Moral,
Self-interest, and Mutual Benefit / Cooperation.

## Participant-level four-category representation exercise

`04_participant_representation_model.py` uses all four HP classifications per
participant-game-condition frame, including `No clear justification` as a fourth category.
It implements a partially pooled version of the proposed multinomial decomposition: HP-allocation
effects are fixed, while participant-frame effects for Moral, Self-interest, and Mutual Benefit /
Cooperation relative to No clear justification are Gaussian-shrunk. The shrinkage penalty is selected
by cross-validation that holds out one HP response per frame in every fold.

The fitted four-category probabilities are averaged over HP anchors 4, 6, 8, and 12 to form
participant-frame weights. The behavioral stage predicts the participant's actual DG share sent from
the Moral, Self-interest, and Cooperation weights, with No clear justification omitted because the four
weights sum to one. It controls for the four game-by-condition cells, clusters uncertainty by Prolific
ID, and reports grouped out-of-sample performance relative to cell controls alone. A participant-cluster
bootstrap refits both stages and checkpoints its draws so it can resume after interruption.
The bootstrap holds the shrinkage penalty at the value selected by full-sample cross-validation.

Run `fit` first, then the resumable bootstrap:

```text
python 04_participant_representation_model.py fit
python 04_participant_representation_model.py bootstrap --reps 500
python 04_participant_representation_model.py status
```

All derived files are written to `participant_representation/`; the round-7 source files are never
modified. The completed bootstrap also regenerates the concise standalone report
`../../hps_weights.tex`.

`05_moral_baseline_excluding_no_clear.py` provides the separate three-category
specification requested after the main analysis. It removes all No-clear HP rows, uses
Moral as the multinomial and behavioral baseline, writes separate checkpointed outputs
to `participant_representation_moral_baseline/`, and adds the results to
`../../hps_weights.tex`.
