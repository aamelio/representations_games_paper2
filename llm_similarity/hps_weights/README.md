# Participant-level HP representation effects

This is the separate workflow behind `../../hps_weights.tex`. It uses the finalized
HP classifications in
`../rounds4_9/hp_similarity/data/hp_responses_classified_and_rated.csv`; similarity
scores are not used in these regressions.

`code/01_participant_representation_model.py` estimates the four-category model
(Moral, Self-interest, Cooperation, No clear justification) and uses the
Gaussian-shrunk participant-frame effects `b_ki` directly to predict DG share
sent, controlling for the four game-by-condition cells. Resumable
participant-cluster bootstrap outputs are written to
`output/four_category_participant_effects/`. Fitted probabilities remain in the
outputs for auditing but are not second-stage predictors.

`code/02_moral_baseline_excluding_no_clear.py` excludes No-clear classifications,
uses Moral as the baseline, and predicts DG share sent directly with the
estimated `b_Si` and `b_Ci`. Its current outputs are written to
`output/moral_baseline_participant_effects/`.

`code/03_between_subject_heterogeneity.py` uses that Moral-baseline sample to
test whether the direct S-versus-M and C-versus-M participant-effect slopes
differ between KW and LT. The Market-versus-Control comparison is restricted to
KW, so its reference cell is KW Control. Its full two-stage bootstrap is
resumable and writes to `output/heterogeneity_participant_effects/`. The scripts
regenerate `../../hps_weights.tex` once their required outputs are complete.

The earlier probability-weight outputs remain in `output/four_category/`,
`output/moral_baseline/`, and `output/heterogeneity/` as an audit trail; they are
not used by the current report generator.

## Within-subject analysis

The within-subject workflow is self-contained under `within_subject/`:

- `input/` contains the long pre-choice HP panel and the deduplicated text list.
- `prompts/classification_prompt.txt` is the frozen M/S/C/N prompt.
- `work/` contains resumable classification checkpoints. Classification uses
  only `classification_id` and `memory`; outcome and treatment fields are not
  shown to the classifiers. The validated combined classifications are in
  `work/validated_rowwise_classifications.csv`. Of the 12,803 newly classified
  unique texts, 12,085 were reviewed by blind GPT-5.6 subagents and the final
  718 were reviewed row by row by the primary agent after the subagent usage
  limit was reached. The file explicitly named
  `rejected_heuristic_classifications.csv` is retained only as an audit record
  and is not used anywhere in the analysis.
- `data/within_hp_panel_classified.csv` is the analysis-ready classified panel.
- `output/` contains the first-stage estimates, participant and treatment
  components, and the period-specific allocation and reasoning models.

`code/04_prepare_within_subject_hp.py` constructs two pre-choice HP elicitation
periods per participant, with four allocation anchors in each period, from the
existing Control KW/LT and Market/Control workbooks. It joins each period to the
allocation and post-choice reasoning observed after that elicitation without
overwriting any source workbook.

`code/05_classify_within_subject_hp.py` is the resumable classification and
import workflow. Exact text matches with the finalized GPT-5.6 between-subject
classification are inherited; all other nonempty texts are classified from the
frozen prompt. Empty text is coded No clear justification. The finalized file
records classification provenance row by row.

`code/06_within_subject_model.py` excludes No-clear HP rows and uses Moral as
the multinomial baseline. It estimates allocation fixed effects, Market, KW,
Market-by-KW, and Gaussian-shrunk participant effects from the pooled first and
second HP elicitations. The current within-subject specification converts the
participant effects into M/S/C probability weights averaged across allocation
anchors at the common LT-Control reference. Treatment components are
the M/S/C probability shifts from LT Control to the current Market/KW cell,
evaluated at zero participant effect. It then asks whether the participant
weights and treatment shifts predict allocation and classified post-choice
reasoning separately in the first and second played game. The post-choice
reasoning labels are the existing labels from
`within/output/within_all_long_categorized.xlsx`, produced by
`within/02_text_categorization.py` with GPT-4.1; this workflow does not
reclassify those outcomes.

Run the within-subject workflow from the repository root:

```text
python representations_games_paper2/llm_similarity/hps_weights/code/04_prepare_within_subject_hp.py
python representations_games_paper2/llm_similarity/hps_weights/code/05_classify_within_subject_hp.py prepare
python representations_games_paper2/llm_similarity/hps_weights/code/05_classify_within_subject_hp.py import --results <validated-classification-csv>
python representations_games_paper2/llm_similarity/hps_weights/code/06_within_subject_model.py fit
python representations_games_paper2/llm_similarity/hps_weights/code/01_participant_representation_model.py tex
```

Both the classifier and model write progress files so an interrupted run can be
resumed or audited. `../../hps_weights.tex` is regenerated from the saved CSV
outputs by `code/01_participant_representation_model.py`.

Run from this directory:

```text
python code/01_participant_representation_model.py fit
python code/01_participant_representation_model.py bootstrap --reps 500
python code/02_moral_baseline_excluding_no_clear.py fit
python code/02_moral_baseline_excluding_no_clear.py bootstrap --reps 500
python code/03_between_subject_heterogeneity.py fit
python code/03_between_subject_heterogeneity.py bootstrap --reps 500
```

Use the corresponding `status` action to inspect a resumable bootstrap.

Python dependencies are listed in `requirements.txt`.
