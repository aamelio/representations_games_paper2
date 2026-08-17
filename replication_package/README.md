# Replication package — "Beyond Social Preferences: Mental Representations in Games"

Amelio, Gennaioli, Nunnari. Not AEA-level; a working package. `../main.tex` draws
every exhibit from this folder (`\graphicspath` → `output/figures/`, `\input` →
`output/tables/`).

## Layout

```
replication_package/
├── README.md
├── data/
│   ├── player{1,2}_all_categorized.xlsx   master categorized workbooks (one row
│   │                                      per response, LLM category)
│   ├── within_switching_results.xlsx      AA's within-subject summary (input to 04)
│   ├── classification_no_clear_justification.xlsx     RA 1 forced labels and
│   ├── control_no_clear_justification_reasons.xlsx    RA 2 forced labels for the
│   │                                      309 unclassified P1 Control/Market answers
│   ├── player1_control_market_no_clear_reclassified.xlsx  the 309 unclassified answers
│   │                                      + the earlier forced LLM pass (all three from
│   │                                      AA's 2026-07-19 email package; inputs to 18)
│   ├── player{1,2}_*_reason_subsample_{1,2}_*.xlsx  6 human-coder validation workbooks
│   ├── within_all_{long,pairs}_categorized.xlsx     within-subject microdata (all 8 from
│   │                                      AA's 2026-07-10 delivery, formerly in
│   │                                      data/missingdata/; inputs to 06)
│   ├── hpmin_social_proximity_all.xlsx    AA's hp (hypothetical-allocation) retained-
│   │                                      memory texts + social-proximity classification
│   │                                      (input to 14; N=1200, 75 PIDs appear in two
│   │                                      cells = repeat participation across DG cells)
│   ├── hpmin_sp_moral_all.xlsx            same + AA's old 7-category moral classification
│   │                                      (column `moral`; AA 2026-07-21; verifies the
│   │                                      app:hp moral exhibits + input to 23)
│   ├── hp_{social_proximity,moral}_all.xlsx  per-level companions (AA 2026-07-21): one
│   │                                      row per participant x hypothetical allocation
│   │                                      (4,800 = 1,200 x 4, hp in {4,6,8,12}); same
│   │                                      participants as the hpmin trio; provenance
│   │                                      record for the retired Table 32/37 splits
│   │                                      (verification/aa_perlevel_checks.py)
│   ├── hpmin_new_scheme_categorized.xlsx  14's deliverable: hp texts reclassified into
│   │                                      the paper's 3-category scheme (input to 15/16)
│   └── memory_games_llm_recording.xlsx    round-1 similarity workbook (input to 05/21)
├── code/
│   ├── 01…23_*.py                         pipeline (run order below)
│   └── verification/                      pure-sympy proposition/remark checks +
│                                          data-side verify_* scripts (round-2
│                                          similarity, app:hp tables 30–37 and
│                                          heatmaps, aa_reply_checks and
│                                          aa_perlevel_checks on AA's 2026-07-21
│                                          answers and per-level files)
└── output/
    ├── figures/                           all paper figures (PNG);
    │   └── fitted_fullpooled/             02's model-fit figures
    ├── unclassified/                      18/19's outputs: reference examples, forced
    │                                      classifications (two models), LOO validation,
    │                                      foldin_summary.txt (every number in the
    │                                      sec:market_control robustness footnote)
    └── tables/                            all paper tables (.tex) + run logs (*.txt);
                                           control_treatment_figure_stats.txt and
                                           forecast_error_figure_stats.txt (2026-07-16)
                                           log every statistic rendered on 01/03's
                                           figures, incl. the all-responses treatment
                                           effects quoted in the paper's text (the
                                           figures use the classified sample; the two
                                           conventions differ by <1pp cell by cell)
```

## Run order

From `code/` (Python 3 with pandas, numpy, matplotlib, statsmodels, scipy, openpyxl):

```
python3 01_control_treatment_and_appendix_figures.py   # control/treatment + appendix figures
python3 02_general_equilibrium_tables_figures.py       # fitted model tables + figures
python3 03_forecast_error_figures.py                   # forecast-error figures + derived CSVs (to data/)
python3 04_paper_v1_extra_outputs.py                   # intro figure, surplus/within tables, Fig. 16
python3 05_paper_v2_new_outputs.py                     # E1-E5 (NOT yet in the paper; pending AA re-run)
python3 06_validation_and_within_checks.py             # LLM-human agreement table + within checks
python3 07_player2_exhibit_split.py                    # P2 figure split (imports 01)
python3 08_calibration.py                              # NG round: per-category (sigma/mu, rho/mu),
                                                       #   (a,b), s; overid tests; Cov(sigma/mu,s)
python3 09_p2_foundation.py                            # NG round: P2 categories vs action faced;
                                                       #   schedules by category; believed-vs-actual
                                                       #   slopes; FE by offer bin
python3 10_interaction_accounting.py                   # NG round: category x belief interactions
python3 11_oaxaca.py                                   # preregistered symmetrized Oaxaca: parallel
                                                       #   category x hypothetical- and actual-belief
                                                       #   cells; matched legacy HP sample; bootstrap SEs B=1000
python3 12_ng_page_items.py                            # NG round: SP x action panel, quote candidates
python3 13_receiver_models.py                          # NG round: receiver protest/equalization fits
                                                       #   (p2_schedules; partly held out of the paper)
python3 14_hp_classification.py                        # NG round: hp memory texts -> new scheme
                                                       #   (Anthropic batch API; --submit/--collect)
python3 15_hp_person_level.py                          # NG round: hp vs reasons person-level table
python3 16_moral_slope_check.py                        # NG round: TG Moral belief-slope diagnostic (held)
python3 17_tg_anchor_dryrun.py                         # NG round: equal-payoff anchor dry run
python3 18_unclassified_classification.py              # forced classification of the 309 unclassified
                                                       #   P1 answers, AA's example-based method
                                                       #   (--build/--classify/--loo; API; headline
                                                       #   model claude-opus-4-8)
python3 19_unclassified_foldin.py                      # fold-in robustness; generates every number in
                                                       #   the sec:market_control unclassified footnote
python3 20_ng_call_prep.py                             # NG call prep: BCS belief terciles, prereg-literal
                                                       #   Oaxaca variants, 4th-category BCS
python3 21_round1_similarity_table.py                  # Table 5 from the round-1 workbook (asserts
                                                       #   vs its Summary sheet)
python3 22_prose_number_backfill.py                    # log backfill: sample counts, TG believer
                                                       #   shares, fold-in max
python3 23_hp_decomposition_tables.py                  # Tables 32/37 (app:hp decompositions) from the
                                                       #   hpmin sample (SP-5 cells, symmetrized)
python3 verification/proof_audit_checks.py             # + the other verification scripts
                                                       #   (proof_*, ng_*, verify_*, aa_*, welfare_*)
```

All scripts anchor paths at the package root via `__file__` (`03` additionally
assumes the working directory is `code/` or the package root).

## Regeneration caveats

- Re-running `02`/`04` rewrites .tex tables with LF line endings where the
  originals were CRLF; content is unchanged but the byte diff is spurious.
- `01` also writes four v1-era figures the paper no longer uses
  (`paper_figure{3,4}_mixed.png`, `paper_outcome_treatment_effects_with_model.png`,
  `paper_representation_treatment_effects.png`); they are deleted from
  `output/figures/` (decision 2026-07-16) and will reappear on a fresh run.
- `05`'s two PNGs re-render with environment-dependent bytes; statistics
  (`paper_v2_new_stats.txt` and all .tex) reproduce exactly (verified 2026-07-16).

## Known gaps

- `output/figures/hp_sp_moral_corr_{ctrl,mkt}.png` (app:hp heatmaps) are AA's
  originals; `verification/verify_hp_moral_tables.py` verifies all 70 annotated
  cells from `data/hpmin_sp_moral_all.xlsx` (AA, 2026-07-21) and writes
  regenerated `*_repro.png` twins. The same script verifies every app:hp table
  (30–37); the Table 35/36 p-values are two-sided equal-proportion tests with
  continuity correction (R `prop.test`), per AA's 2026-07-21 reply (the three
  numeric entries were corrected accordingly, see
  `verification/aa_reply_checks.py`). The former sole remainder — the
  representation/behavior splits of Tables 32/37 — is CLOSED: AA recovered his
  code and confirmed the published splits came from an erroneous
  all-observations run (all four hp texts per participant instead of the
  retained-text sample); the tables are now generated by
  `code/23_hp_decomposition_tables.py` (hpmin sample, SP-5 cells, symmetrized,
  matching the captions) and fully verified. AA's per-level files
  (`data/hp_{social_proximity,moral}_all.xlsx`) are validated against the hpmin
  trio in `verification/aa_perlevel_checks.py`, which also documents that the
  retired published splits match no standard construction even on the
  per-level data (72 constructions per outcome row).
- `output/tables/within_*regression_table*.tex` (4 files) are AA's, not
  script-generated; edited 2026-07-10 to drop the non-identified LTFirst/ControlFirst
  rows (originals in `../backups/within_tables_2026-07-10/`). `06` reproduces every
  FE-identified coefficient from `data/within_all_long_categorized.xlsx`.
- `output/figures/choice_switch_control_kw_first.png` is AA's original of Fig. 16;
  the paper uses `04`'s regenerated `within_choice_switch_kw_first.png`.
- `05`'s similarity constants (94.6/51.2/29.9/14.4) are audited means from
  `../LLM_Similarity/memory_games_llm_recording.xlsx`, hard-coded, not read at runtime.
