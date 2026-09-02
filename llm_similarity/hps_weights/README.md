# Participant-level HP representation weights

This is the separate workflow behind `../../hps_weights.tex`. It uses the finalized
HP classifications in
`../rounds4_9/hp_similarity/data/hp_responses_classified_and_rated.csv`; similarity
scores are not used in these regressions.

`code/01_participant_representation_model.py` estimates the four-category model
(Moral, Self-interest, Cooperation, No clear justification), predicts DG share
sent with game-by-condition controls, and writes resumable participant-cluster
bootstrap outputs to `output/four_category/`.

`code/02_moral_baseline_excluding_no_clear.py` excludes No-clear classifications,
uses Moral as the baseline, and writes its outputs to `output/moral_baseline/`.
Both specifications regenerate `../../hps_weights.tex` once their required outputs
are complete.

Run from this directory:

```text
python code/01_participant_representation_model.py fit
python code/01_participant_representation_model.py bootstrap --reps 500
python code/02_moral_baseline_excluding_no_clear.py fit
python code/02_moral_baseline_excluding_no_clear.py bootstrap --reps 500
```

Use the corresponding `status` action to inspect a resumable bootstrap.

Python dependencies are listed in `requirements.txt`.

