"""Cell-specific within-subject HP models and pooled outcome predictions.

No-clear HP classifications are excluded and Moral is the multinomial
reference category.  The first-stage model is estimated separately in each
of the four game-by-condition cells.  The resulting fitted M/S/C weights are
then used in pooled allocation and post-choice reasoning models.

All outputs are written to ``within_subject/output/cell_specific`` so the
earlier unified within-subject specification remains available for audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
WITHIN = ROOT / "within_subject"
INPUT = WITHIN / "data" / "within_hp_panel_classified.csv"
OUT = WITHIN / "output" / "cell_specific"
PROGRESS = OUT / "progress.json"

CELL_ORDER = ["kw_control", "lt_control", "kw_market", "lt_market"]
CELL_LABELS = {
    "kw_control": "KW Control",
    "lt_control": "LT Control",
    "kw_market": "KW Market",
    "lt_market": "LT Market",
}
CATEGORIES = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
MODELED = CATEGORIES[1:]
SHORT = {
    "Moral": "M",
    "Self-interest": "S",
    "Mutual Benefit / Cooperation": "C",
}
ANCHORS = [4.0, 6.0, 8.0, 12.0]
LAMBDA_GRID = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def load_base_module():
    path = Path(__file__).with_name("06_within_subject_model.py")
    spec = importlib.util.spec_from_file_location("within_unified_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def write_json(path: Path, value: dict) -> None:
    BASE.atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    full = pd.read_csv(INPUT)
    full["subject_id"] = full["subject_id"].astype(str)
    full["PROLIFIC_PID"] = full["PROLIFIC_PID"].astype(str)
    full["anchor_allocation_kept"] = pd.to_numeric(
        full["anchor_allocation_kept"], errors="raise"
    )
    full["actual_share_sent"] = pd.to_numeric(
        full["actual_share_sent"], errors="raise"
    )
    full["trial_order"] = pd.to_numeric(
        full["trial_order"], errors="raise"
    ).astype(int)
    substantive = full[full["category"].isin(CATEGORIES)].copy()
    elicitation = full.drop_duplicates(["subject_id", "trial_order"])
    if len(elicitation) != full["subject_id"].nunique() * 2:
        raise AssertionError("Expected exactly two elicitations per participant.")
    if elicitation.groupby(["subject_id", "treatment_cell"]).size().max() != 1:
        raise AssertionError("A participant must appear at most once in a cell.")
    summary = {
        "n_subjects_all": int(full["subject_id"].nunique()),
        "n_elicitations_all": int(len(elicitation)),
        "n_hp_rows_all": int(len(full)),
        "n_hp_rows_substantive": int(len(substantive)),
        "n_hp_rows_no_clear": int(len(full) - len(substantive)),
    }
    return full, substantive, summary


def fixed_design(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    anchor = data["anchor_allocation_kept"].astype(float)
    names = ["Intercept", "Anchor=6", "Anchor=8", "Anchor=12"]
    design = np.column_stack([
        np.ones(len(data)),
        (anchor == 6).astype(float),
        (anchor == 8).astype(float),
        (anchor == 12).astype(float),
    ])
    return design, names


def prepare_first_stage(data: pd.DataFrame):
    groups = sorted(data["subject_id"].unique())
    group_lookup = {group: index for index, group in enumerate(groups)}
    group_codes = data["subject_id"].map(group_lookup).to_numpy(dtype=int)
    design, fixed_names = fixed_design(data)
    category_lookup = {category: index for index, category in enumerate(CATEGORIES)}
    outcome = data["category"].map(category_lookup).to_numpy(dtype=int)
    return design, group_codes, outcome, fixed_names, groups


def make_cv_folds(data: pd.DataFrame) -> np.ndarray:
    subject_order = {
        subject: index
        for index, subject in enumerate(sorted(data["subject_id"].unique()))
    }
    anchor_order = {anchor: index for index, anchor in enumerate(ANCHORS)}
    subject_index = data["subject_id"].map(subject_order).to_numpy(dtype=int)
    anchor_index = data["anchor_allocation_kept"].map(anchor_order).to_numpy(dtype=int)
    return (subject_index + anchor_index) % 4


def tune_penalty(data: pd.DataFrame, cell: str) -> pd.DataFrame:
    folds = make_cv_folds(data)
    records: list[dict] = []
    for fold in range(4):
        train = data.loc[folds != fold].copy()
        test = data.loc[folds == fold].copy()
        train_x, train_groups, train_y, _, group_names = prepare_first_stage(train)
        group_lookup = {group: index for index, group in enumerate(group_names)}
        test_groups = test["subject_id"].map(group_lookup).fillna(-1).to_numpy(dtype=int)
        test_x, _ = fixed_design(test)
        category_lookup = {category: index for index, category in enumerate(CATEGORIES)}
        test_y = test["category"].map(category_lookup).to_numpy(dtype=int)
        initial = None
        for penalty in sorted(LAMBDA_GRID, reverse=True):
            fit = BASE.fit_penalized_multinomial(
                train_x,
                train_groups,
                train_y,
                penalty,
                initial=initial,
            )
            initial = np.concatenate([fit.fixed.ravel(), fit.participant.ravel()])
            probability = BASE.predict_probabilities(fit, test_x, test_groups)
            chosen = probability[np.arange(len(test_y)), test_y]
            records.append({
                "treatment_cell": cell,
                "fold": fold + 1,
                "penalty_lambda": penalty,
                "n_train": len(train),
                "n_test": len(test),
                "n_unseen_test_rows": int((test_groups < 0).sum()),
                "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-15, 1.0)))),
                "accuracy": float((probability.argmax(axis=1) == test_y).mean()),
                "converged": fit.success,
                "iterations": fit.iterations,
                "gradient_max": fit.gradient_max,
            })
    result = pd.DataFrame(records)
    means = result.groupby("penalty_lambda", observed=True)["log_loss"].mean()
    selected = float(means.idxmin())
    result["mean_log_loss"] = result["penalty_lambda"].map(means)
    result["selected"] = result["penalty_lambda"].eq(selected)
    return result.sort_values(["penalty_lambda", "fold"]).reset_index(drop=True)


def fit_cell(
    full: pd.DataFrame, substantive: pd.DataFrame, cell: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cell_all = full[full["treatment_cell"] == cell].copy()
    cell_data = substantive[substantive["treatment_cell"] == cell].copy()
    cv = tune_penalty(cell_data, cell)
    selected = float(cv.loc[cv["selected"], "penalty_lambda"].iloc[0])
    design, groups, outcome, fixed_names, group_names = prepare_first_stage(cell_data)
    fit = BASE.fit_penalized_multinomial(design, groups, outcome, selected)

    fixed_rows = []
    for fixed_index, term in enumerate(fixed_names):
        for category_index, category in enumerate(MODELED):
            fixed_rows.append({
                "treatment_cell": cell,
                "term": term,
                "category_vs_moral": SHORT[category],
                "log_odds_coefficient": fit.fixed[fixed_index, category_index],
                "odds_ratio": np.exp(fit.fixed[fixed_index, category_index]),
            })
    fixed = pd.DataFrame(fixed_rows)

    participant = pd.DataFrame({
        "subject_id": group_names,
        "treatment_cell": cell,
        "b_S_vs_M": fit.participant[:, 0],
        "b_C_vs_M": fit.participant[:, 1],
    })
    counts = cell_data.groupby("subject_id").size().rename("n_substantive_hp")
    participant = participant.merge(
        counts, left_on="subject_id", right_index=True, how="left", validate="one_to_one"
    )

    grid = participant[["subject_id"]].assign(_key=1).merge(
        pd.DataFrame({"anchor_allocation_kept": ANCHORS, "_key": 1}),
        on="_key",
        how="inner",
    ).drop(columns="_key")
    grid_design, _ = fixed_design(grid)
    group_lookup = {group: index for index, group in enumerate(group_names)}
    group_codes = grid["subject_id"].map(group_lookup).to_numpy(dtype=int)
    probability = BASE.predict_probabilities(fit, grid_design, group_codes)
    for category_index, category in enumerate(["M", "S", "C"]):
        grid[f"weight_{category}"] = probability[:, category_index]
    average_weights = grid.groupby("subject_id", as_index=False)[
        ["weight_M", "weight_S", "weight_C"]
    ].mean()
    participant = participant.merge(
        average_weights, on="subject_id", how="left", validate="one_to_one"
    )
    if not np.allclose(
        participant[["weight_M", "weight_S", "weight_C"]].sum(axis=1), 1.0
    ):
        raise AssertionError("M/S/C weights must sum to one.")

    metadata_columns = [
        "subject_id",
        "PROLIFIC_PID",
        "source_study",
        "game",
        "Market",
        "kw_dummy",
        "treatment_cell",
        "trial_order",
        "actual_allocation_kept",
        "actual_share_sent",
        "actual_reason",
        "actual_reason_category",
        "actual_reason_category_num",
    ]
    metadata = cell_all[metadata_columns].drop_duplicates(["subject_id", "treatment_cell"])
    weights = metadata.merge(
        participant,
        on=["subject_id", "treatment_cell"],
        how="inner",
        validate="one_to_one",
    )
    summary = {
        "treatment_cell": cell,
        "n_elicitations_all": int(cell_all["subject_id"].nunique()),
        "n_elicitations_usable": int(weights["subject_id"].nunique()),
        "n_elicitations_all_no_clear": int(
            cell_all["subject_id"].nunique() - weights["subject_id"].nunique()
        ),
        "n_hp_rows_all": int(len(cell_all)),
        "n_hp_rows_substantive": int(len(cell_data)),
        "selected_lambda": selected,
        "converged": bool(fit.success),
        "iterations": int(fit.iterations),
        "gradient_max": float(fit.gradient_max),
    }
    return cv, fixed, participant, weights, summary


def pooled_design(data: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame({
        "const": 1.0,
        "weight_S": data["weight_S"].astype(float),
        "weight_C": data["weight_C"].astype(float),
        "LT_Control": (data["treatment_cell"] == "lt_control").astype(float),
        "KW_Market": (data["treatment_cell"] == "kw_market").astype(float),
        "LT_Market": (data["treatment_cell"] == "lt_market").astype(float),
        "second_trial": (data["trial_order"] == 2).astype(float),
        "market_control_study": (data["source_study"] == "market_control").astype(float),
    }, index=data.index)
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise AssertionError("The pooled design matrix is not full rank.")
    return design


def pooled_allocation_model(weights: pd.DataFrame):
    data = weights.copy()
    design = pooled_design(data)
    model = sm.OLS(data["actual_share_sent"], design).fit(
        cov_type="cluster", cov_kwds={"groups": data["PROLIFIC_PID"]}
    )
    confidence = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append({
            "term": term,
            "coefficient": model.params[term],
            "std_error_clustered_pid": model.bse[term],
            "p_value": model.pvalues[term],
            "ci95_low": confidence.loc[term, 0],
            "ci95_high": confidence.loc[term, 1],
            "n": int(model.nobs),
            "n_participants": int(data["PROLIFIC_PID"].nunique()),
            "r_squared": model.rsquared,
        })
    predictions = data[[
        "subject_id", "PROLIFIC_PID", "source_study", "treatment_cell",
        "trial_order", "actual_share_sent", "weight_M", "weight_S", "weight_C",
    ]].copy()
    predictions["fitted_share_sent"] = model.predict(design)
    predictions["residual"] = predictions["actual_share_sent"] - predictions["fitted_share_sent"]
    return model, pd.DataFrame(rows), predictions


def pooled_reasoning_model(weights: pd.DataFrame):
    data = weights[weights["actual_reason_category"].isin(CATEGORIES)].copy()
    design = pooled_design(data)
    outcome = data["actual_reason_category"].map(
        {category: index for index, category in enumerate(CATEGORIES)}
    ).astype(int)
    model = sm.MNLogit(outcome, design).fit(
        method="newton",
        maxiter=200,
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data["PROLIFIC_PID"]},
    )
    rows = []
    for outcome_column, outcome_category in enumerate(["S", "C"]):
        for term in model.params.index:
            coefficient = model.params.loc[term, outcome_column]
            standard_error = model.bse.loc[term, outcome_column]
            rows.append({
                "outcome_vs_moral": outcome_category,
                "term": term,
                "coefficient": coefficient,
                "std_error_clustered_pid": standard_error,
                "p_value": model.pvalues.loc[term, outcome_column],
                "ci95_low": coefficient - 1.959963984540054 * standard_error,
                "ci95_high": coefficient + 1.959963984540054 * standard_error,
                "n": int(model.nobs),
                "n_participants": int(data["PROLIFIC_PID"].nunique()),
                "pseudo_r_squared": float(model.prsquared),
            })
    probability = np.asarray(model.predict(design))
    outcome_array = outcome.to_numpy(dtype=int)
    performance = pd.DataFrame([{
        "n": int(model.nobs),
        "n_participants": int(data["PROLIFIC_PID"].nunique()),
        "log_loss": float(
            -np.mean(np.log(np.clip(probability[np.arange(len(data)), outcome_array], 1e-15, 1.0)))
        ),
        "accuracy": float((probability.argmax(axis=1) == outcome_array).mean()),
        "pseudo_r_squared": float(model.prsquared),
    }])
    return pd.DataFrame(rows), performance


def paired_comparisons(weights: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("KW Control to LT Control", "kw_control", "lt_control"),
        ("KW Control to KW Market", "kw_control", "kw_market"),
    ]
    rows = []
    columns = [
        "PROLIFIC_PID", "weight_M", "weight_S", "weight_C",
        "actual_share_sent", "trial_order",
    ]
    for label, origin, destination in definitions:
        left = weights[weights["treatment_cell"] == origin][columns].copy()
        right = weights[weights["treatment_cell"] == destination][columns].copy()
        paired = left.merge(
            right,
            on="PROLIFIC_PID",
            suffixes=("_origin", "_destination"),
            how="inner",
            validate="one_to_one",
        )
        for _, row in paired.iterrows():
            rows.append({
                "comparison": label,
                "origin_cell": origin,
                "destination_cell": destination,
                "PROLIFIC_PID": row["PROLIFIC_PID"],
                "origin_trial_order": int(row["trial_order_origin"]),
                "destination_trial_order": int(row["trial_order_destination"]),
                "origin_weight_M": row["weight_M_origin"],
                "origin_weight_S": row["weight_S_origin"],
                "origin_weight_C": row["weight_C_origin"],
                "destination_weight_M": row["weight_M_destination"],
                "destination_weight_S": row["weight_S_destination"],
                "destination_weight_C": row["weight_C_destination"],
                "delta_weight_M": row["weight_M_destination"] - row["weight_M_origin"],
                "delta_weight_S": row["weight_S_destination"] - row["weight_S_origin"],
                "delta_weight_C": row["weight_C_destination"] - row["weight_C_origin"],
                "origin_share_sent": row["actual_share_sent_origin"],
                "destination_share_sent": row["actual_share_sent_destination"],
                "delta_share_sent": row["actual_share_sent_destination"] - row["actual_share_sent_origin"],
            })
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(0.975, n - 1))
    return mean, se, mean - critical * se, mean + critical * se


def treatment_shift_decomposition(
    paired: pd.DataFrame, allocation_model
) -> pd.DataFrame:
    beta_s = float(allocation_model.params["weight_S"])
    beta_c = float(allocation_model.params["weight_C"])
    rows = []
    for comparison, data in paired.groupby("comparison", sort=False):
        predicted = beta_s * data["delta_weight_S"] + beta_c * data["delta_weight_C"]
        predicted_mean, predicted_se, predicted_low, predicted_high = mean_ci(predicted)
        actual_mean, actual_se, actual_low, actual_high = mean_ci(data["delta_share_sent"])
        rows.append({
            "comparison": comparison,
            "n_paired_participants": len(data),
            "mean_delta_weight_M": data["delta_weight_M"].mean(),
            "mean_delta_weight_S": data["delta_weight_S"].mean(),
            "mean_delta_weight_C": data["delta_weight_C"].mean(),
            "beta_S": beta_s,
            "beta_C": beta_c,
            "predicted_allocation_shift_from_weights": predicted_mean,
            "predicted_shift_se": predicted_se,
            "predicted_shift_ci95_low": predicted_low,
            "predicted_shift_ci95_high": predicted_high,
            "actual_mean_allocation_shift": actual_mean,
            "actual_shift_se": actual_se,
            "actual_shift_ci95_low": actual_low,
            "actual_shift_ci95_high": actual_high,
        })
    return pd.DataFrame(rows)


def moral_weight_transitions(
    weights: pd.DataFrame, paired: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = weights.loc[
        weights["treatment_cell"] == "kw_control", "weight_M"
    ]
    q1, q2 = reference.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    labels = ["Low", "Middle", "High"]

    def assign(values: pd.Series) -> pd.Categorical:
        result = np.where(values <= q1, "Low", np.where(values <= q2, "Middle", "High"))
        return pd.Categorical(result, categories=labels, ordered=True)

    rows = []
    for comparison, data in paired.groupby("comparison", sort=False):
        frame = pd.DataFrame({
            "origin_tercile": assign(data["origin_weight_M"]),
            "destination_tercile": assign(data["destination_weight_M"]),
        })
        table = pd.crosstab(
            frame["origin_tercile"], frame["destination_tercile"], dropna=False
        ).reindex(index=labels, columns=labels, fill_value=0)
        for origin in labels:
            row_n = int(table.loc[origin].sum())
            for destination in labels:
                count = int(table.loc[origin, destination])
                rows.append({
                    "comparison": comparison,
                    "origin_tercile": origin,
                    "destination_tercile": destination,
                    "count": count,
                    "row_n": row_n,
                    "row_percent": 100.0 * count / row_n if row_n else np.nan,
                })
    cutoffs = pd.DataFrame([{
        "reference_cell": "kw_control",
        "reference_n": len(reference),
        "lower_cutoff": q1,
        "upper_cutoff": q2,
    }])
    return pd.DataFrame(rows), cutoffs


def run_fit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, substantive, summary = load_data()
    all_cv = []
    all_fixed = []
    all_participant = []
    all_weights = []
    cell_summaries = []
    write_json(PROGRESS, {"stage": "first_stage", "completed_cells": []})
    for cell in CELL_ORDER:
        cv, fixed, participant, weights, cell_summary = fit_cell(
            full, substantive, cell
        )
        all_cv.append(cv)
        all_fixed.append(fixed)
        all_participant.append(participant)
        all_weights.append(weights)
        cell_summaries.append(cell_summary)
        BASE.atomic_write_csv(cv, OUT / f"first_stage_lambda_cv_{cell}.csv")
        BASE.atomic_write_csv(fixed, OUT / f"first_stage_fixed_effects_{cell}.csv")
        BASE.atomic_write_csv(participant, OUT / f"participant_effects_{cell}.csv")
        BASE.atomic_write_csv(weights, OUT / f"participant_weights_{cell}.csv")
        write_json(PROGRESS, {
            "stage": "first_stage",
            "completed_cells": [item["treatment_cell"] for item in cell_summaries],
        })

    cv = pd.concat(all_cv, ignore_index=True)
    fixed = pd.concat(all_fixed, ignore_index=True)
    participant = pd.concat(all_participant, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    cell_means = weights.groupby("treatment_cell", as_index=False)[
        ["weight_M", "weight_S", "weight_C"]
    ].mean()

    allocation_model, allocation, allocation_predictions = pooled_allocation_model(weights)
    reasoning, reasoning_performance = pooled_reasoning_model(weights)
    paired = paired_comparisons(weights)
    decomposition = treatment_shift_decomposition(paired, allocation_model)
    transitions, cutoffs = moral_weight_transitions(weights, paired)

    BASE.atomic_write_csv(cv, OUT / "first_stage_lambda_cv.csv")
    BASE.atomic_write_csv(fixed, OUT / "first_stage_fixed_effects.csv")
    BASE.atomic_write_csv(participant, OUT / "participant_cell_effects.csv")
    BASE.atomic_write_csv(weights, OUT / "participant_cell_weights.csv")
    BASE.atomic_write_csv(cell_means, OUT / "cell_weight_means.csv")
    BASE.atomic_write_csv(allocation, OUT / "pooled_allocation_model.csv")
    BASE.atomic_write_csv(allocation_predictions, OUT / "pooled_allocation_predictions.csv")
    BASE.atomic_write_csv(reasoning, OUT / "pooled_reasoning_model.csv")
    BASE.atomic_write_csv(reasoning_performance, OUT / "pooled_reasoning_performance.csv")
    BASE.atomic_write_csv(paired, OUT / "paired_weight_shifts.csv")
    BASE.atomic_write_csv(decomposition, OUT / "treatment_shift_decomposition.csv")
    BASE.atomic_write_csv(transitions, OUT / "moral_weight_transition_matrices.csv")
    BASE.atomic_write_csv(cutoffs, OUT / "moral_weight_tercile_cutoffs.csv")

    summary.update({
        "n_usable_participant_cells": int(len(weights)),
        "n_unique_usable_participants": int(weights["PROLIFIC_PID"].nunique()),
        "allocation_model_n": int(allocation_model.nobs),
        "allocation_model_r_squared": float(allocation_model.rsquared),
        "cells": cell_summaries,
        "moral_weight_tercile_reference": "KW Control",
        "moral_weight_tercile_lower_cutoff": float(cutoffs.loc[0, "lower_cutoff"]),
        "moral_weight_tercile_upper_cutoff": float(cutoffs.loc[0, "upper_cutoff"]),
    })
    write_json(OUT / "sample_summary.json", summary)
    write_json(PROGRESS, {
        "stage": "complete",
        "fit_complete": True,
        "analysis_complete": True,
        **summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


def show_status() -> None:
    if not PROGRESS.exists():
        print("Cell-specific within-subject analysis has not been started.")
        return
    print(PROGRESS.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "status"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "fit":
        run_fit()
    else:
        show_status()


if __name__ == "__main__":
    main()
