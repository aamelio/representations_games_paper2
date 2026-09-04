#!/usr/bin/env python3
"""Estimate participant-level HP category effects and predict DG choices.

The first stage implements a partially pooled four-category multinomial logit:

    log Pr(K_ia = k) / Pr(K_ia = N)
        = allocation_effect[k, a] + participant_frame_effect[k, i],

where k is Moral (M), Self-interest (S), or Cooperation (C), and N is
"No clear justification". Participant-frame effects receive an L2 penalty,
equivalent to a zero-mean Gaussian prior in a maximum-a-posteriori fit. The
penalty is selected by four-fold cross-validation that holds out one of each
frame's four HP responses in every fold.

The second stage uses the estimated participant-frame effects directly. It
regresses the participant's actual DG share sent on the M-, S-, and C-versus-N
effects plus game-by-condition fixed effects, with standard errors clustered by
Prolific ID. Fitted category probabilities are retained only as audit outputs.

All outputs go under output/four_category_participant_effects/. The optional participant-
cluster bootstrap refits both stages and checkpoints every few replicates.

Usage:
    python 01_participant_representation_model.py fit
    python 01_participant_representation_model.py bootstrap --reps 500
    python 01_participant_representation_model.py all --reps 500
    python 01_participant_representation_model.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT.parent / "rounds4_9" / "hp_similarity" / "data"
    / "hp_responses_classified_and_rated.csv"
)
OUT = ROOT / "output" / "four_category_participant_effects"
PROGRESS = OUT / "progress.json"
TEX_OUTPUT = ROOT.parent.parent / "hps_weights.tex"

BASE_CATEGORY = "No clear justification"
MODELED_CATEGORIES = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
CATEGORIES = [BASE_CATEGORY, *MODELED_CATEGORIES]
CATEGORY_SHORT = {
    BASE_CATEGORY: "N",
    "Moral": "M",
    "Self-interest": "S",
    "Mutual Benefit / Cooperation": "C",
}
HP_LEVELS = [4.0, 6.0, 8.0, 12.0]
LAMBDA_GRID_DEFAULT = [
    0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
    8.0, 16.0, 32.0,
]
BOOTSTRAP_SEED = 20260824


@dataclass
class FirstStageFit:
    fixed: np.ndarray
    participant: np.ndarray
    success: bool
    iterations: int
    objective: float
    gradient_max: float
    message: str


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(source_hash: str, lambda_grid: Iterable[float]) -> str:
    specification = {
        "source_sha256": source_hash,
        "base_category": BASE_CATEGORY,
        "modeled_categories": MODELED_CATEGORIES,
        "hp_levels": HP_LEVELS,
        "lambda_grid": list(lambda_grid),
        "participant_unit": "PROLIFIC_PID x treatment x Market",
        "second_stage": "share_sent on direct M/S/C-versus-N participant effects plus game-by-condition FE",
    }
    payload = json.dumps(specification, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_progress() -> dict:
    if not PROGRESS.exists():
        return {}
    return json.loads(PROGRESS.read_text(encoding="utf-8"))


def write_progress(**updates: object) -> dict:
    current = read_progress()
    current.update(updates)
    atomic_write_text(PROGRESS, json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {
        "hp_response_id", "PROLIFIC_PID", "treatment", "Market", "allocation",
        "hp", "category_num", "category",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        bad = data[list(required)].isna().sum()
        raise ValueError(f"Missing required values:\n{bad[bad > 0]}")
    if set(data["category"]) != set(CATEGORIES):
        raise ValueError(
            f"Expected exactly the four categories {CATEGORIES}; "
            f"found {sorted(data['category'].unique())}"
        )
    if set(data["hp"].astype(float)) != set(HP_LEVELS):
        raise ValueError(f"Unexpected HP anchors: {sorted(data['hp'].unique())}")
    if not data["treatment"].isin(["kw", "lt"]).all():
        raise ValueError("Treatment must be kw or lt.")
    if not data["Market"].isin([0, 1]).all():
        raise ValueError("Market must be coded 0/1.")

    data = data.copy()
    data["frame_id"] = (
        data["PROLIFIC_PID"].astype(str) + "|" + data["treatment"].astype(str)
        + "|" + data["Market"].astype(str)
    )
    counts = data.groupby("frame_id", observed=True).size()
    if len(data) != 4800 or len(counts) != 1200 or not counts.eq(4).all():
        raise ValueError(
            "Expected 4,800 rows in 1,200 four-response participant-frames; "
            f"found {len(data)} rows and {len(counts)} frames with counts "
            f"{counts.value_counts().sort_index().to_dict()}."
        )
    anchors = data.groupby("frame_id", observed=True)["hp"].apply(
        lambda values: tuple(sorted(values.astype(float)))
    )
    if not anchors.map(lambda value: value == tuple(HP_LEVELS)).all():
        raise ValueError("Every frame must contain HP=4, 6, 8, and 12 exactly once.")
    invariant = data.groupby("frame_id", observed=True).agg(
        pid=("PROLIFIC_PID", "nunique"),
        game=("treatment", "nunique"),
        condition=("Market", "nunique"),
        actual_allocation=("allocation", "nunique"),
    )
    if not invariant.eq(1).all().all():
        raise ValueError("Participant ID, cell, or actual allocation varies within frame.")
    expected_numbers = {
        "No clear justification": 0,
        "Moral": 1,
        "Mutual Benefit / Cooperation": 2,
        "Self-interest": 3,
    }
    for row in data[["category", "category_num"]].drop_duplicates().itertuples(index=False):
        if expected_numbers[row.category] != int(row.category_num):
            raise ValueError(f"Unexpected category coding: {row.category}={row.category_num}")
    return data


def prepare_first_stage(
    data: pd.DataFrame,
    group_column: str = "frame_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    ordered_groups = sorted(data[group_column].astype(str).unique())
    group_lookup = {value: index for index, value in enumerate(ordered_groups)}
    group_codes = data[group_column].astype(str).map(group_lookup).to_numpy(dtype=int)
    hp = data["hp"].astype(float)
    fixed_names = ["Intercept", "HP=6", "HP=8", "HP=12"]
    fixed = np.column_stack([
        np.ones(len(data)),
        (hp == 6).astype(float),
        (hp == 8).astype(float),
        (hp == 12).astype(float),
    ])
    category_lookup = {category: code for code, category in enumerate(CATEGORIES)}
    outcome = data["category"].map(category_lookup).to_numpy(dtype=int)
    return fixed, group_codes, outcome, fixed_names, ordered_groups


def fit_penalized_multinomial(
    fixed_design: np.ndarray,
    group_codes: np.ndarray,
    outcome: np.ndarray,
    penalty: float,
    initial: np.ndarray | None = None,
    maxiter: int = 500,
) -> FirstStageFit:
    n_fixed = fixed_design.shape[1]
    n_groups = int(group_codes.max()) + 1
    n_logits = len(MODELED_CATEGORIES)
    parameter_count = n_fixed * n_logits + n_groups * n_logits
    if initial is None or len(initial) != parameter_count:
        initial = np.zeros(parameter_count, dtype=float)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split = n_fixed * n_logits
        return (
            parameters[:split].reshape(n_fixed, n_logits),
            parameters[split:].reshape(n_groups, n_logits),
        )

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        fixed_parameters, participant_parameters = unpack(parameters)
        nonbase = fixed_design @ fixed_parameters + participant_parameters[group_codes]
        logits = np.column_stack([np.zeros(len(outcome)), nonbase])
        log_denominator = logsumexp(logits, axis=1)
        objective = float(np.sum(log_denominator - logits[np.arange(len(outcome)), outcome]))
        objective += 0.5 * penalty * float(np.sum(participant_parameters ** 2))
        probabilities = np.exp(logits - log_denominator[:, None])
        residual = probabilities[:, 1:]
        for category_code in range(1, len(CATEGORIES)):
            residual[:, category_code - 1] -= outcome == category_code
        fixed_gradient = fixed_design.T @ residual
        participant_gradient = np.zeros_like(participant_parameters)
        np.add.at(participant_gradient, group_codes, residual)
        participant_gradient += penalty * participant_parameters
        gradient = np.concatenate([fixed_gradient.ravel(), participant_gradient.ravel()])
        return objective, gradient

    result = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-6, "maxls": 40},
    )
    fixed_parameters, participant_parameters = unpack(result.x)
    return FirstStageFit(
        fixed=fixed_parameters,
        participant=participant_parameters,
        success=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
        gradient_max=float(np.max(np.abs(result.jac))),
        message=str(result.message),
    )


def predict_probabilities(
    fit: FirstStageFit,
    fixed_design: np.ndarray,
    group_codes: np.ndarray,
) -> np.ndarray:
    nonbase = fixed_design @ fit.fixed + fit.participant[group_codes]
    logits = np.column_stack([np.zeros(len(fixed_design)), nonbase])
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def make_cv_folds(data: pd.DataFrame) -> np.ndarray:
    frame_order = {frame: i for i, frame in enumerate(sorted(data["frame_id"].unique()))}
    hp_order = {hp: i for i, hp in enumerate(HP_LEVELS)}
    frame_index = data["frame_id"].map(frame_order).to_numpy(dtype=int)
    hp_index = data["hp"].astype(float).map(hp_order).to_numpy(dtype=int)
    folds = (frame_index + hp_index) % 4
    fold_counts = pd.crosstab(data["frame_id"], folds)
    if fold_counts.shape != (1200, 4) or not fold_counts.eq(1).all().all():
        raise AssertionError("CV must hold out one response per frame in every fold.")
    return folds


def tune_penalty(data: pd.DataFrame, lambda_grid: list[float]) -> pd.DataFrame:
    folds = make_cv_folds(data)
    records: list[dict] = []
    for fold in range(4):
        train = data.loc[folds != fold].copy()
        test = data.loc[folds == fold].copy()
        train_x, train_groups, train_y, _, group_names = prepare_first_stage(train)
        group_lookup = {name: index for index, name in enumerate(group_names)}
        test_groups = test["frame_id"].map(group_lookup).to_numpy(dtype=int)
        test_x, _, test_y, _, _ = prepare_first_stage(test)
        initial = None
        for penalty in sorted(lambda_grid, reverse=True):
            fit = fit_penalized_multinomial(
                train_x, train_groups, train_y, penalty, initial=initial
            )
            initial = np.concatenate([fit.fixed.ravel(), fit.participant.ravel()])
            probabilities = predict_probabilities(fit, test_x, test_groups)
            selected = probabilities[np.arange(len(test_y)), test_y]
            records.append({
                "fold": fold + 1,
                "penalty_lambda": penalty,
                "n_train": len(train),
                "n_test": len(test),
                "log_loss": float(-np.mean(np.log(np.clip(selected, 1e-15, 1)))),
                "accuracy": float((probabilities.argmax(axis=1) == test_y).mean()),
                "converged": fit.success,
                "iterations": fit.iterations,
                "gradient_max": fit.gradient_max,
                "message": fit.message,
            })
    result = pd.DataFrame(records)
    result["mean_log_loss"] = result.groupby("penalty_lambda", observed=True)[
        "log_loss"
    ].transform("mean")
    selected = result.groupby("penalty_lambda", observed=True)["log_loss"].mean().idxmin()
    result["selected"] = result["penalty_lambda"].eq(float(selected))
    return result.sort_values(["penalty_lambda", "fold"]).reset_index(drop=True)


def fixed_effect_table(fit: FirstStageFit, fixed_names: list[str]) -> pd.DataFrame:
    records = []
    for fixed_index, term in enumerate(fixed_names):
        for category_index, category in enumerate(MODELED_CATEGORIES):
            records.append({
                "term": term,
                "category_vs_no_clear": category,
                "log_odds_coefficient": fit.fixed[fixed_index, category_index],
                "odds_ratio": np.exp(fit.fixed[fixed_index, category_index]),
            })
    return pd.DataFrame(records)


def build_fitted_outputs(
    data: pd.DataFrame,
    fit: FirstStageFit,
    fixed_design: np.ndarray,
    group_codes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = predict_probabilities(fit, fixed_design, group_codes)
    row_output = data[[
        "hp_response_id", "frame_id", "PROLIFIC_PID", "treatment", "Market",
        "allocation", "hp", "category_num", "category",
    ]].copy()
    for category_index, category in enumerate(CATEGORIES):
        row_output[f"fitted_probability_{CATEGORY_SHORT[category]}"] = probabilities[:, category_index]
    effect_columns = []
    for category_index, category in enumerate(MODELED_CATEGORIES):
        column = f"b_{CATEGORY_SHORT[category]}_vs_N"
        effect_columns.append(column)
        row_output[column] = fit.participant[group_codes, category_index]

    metadata = data.groupby("frame_id", as_index=False, observed=True).agg(
        PROLIFIC_PID=("PROLIFIC_PID", "first"),
        treatment=("treatment", "first"),
        Market=("Market", "first"),
        allocation_kept=("allocation", "first"),
    )
    metadata["share_sent"] = (12.0 - metadata["allocation_kept"]) / 12.0
    metadata["cell"] = metadata["treatment"] + "_" + metadata["Market"].map(
        {0: "control", 1: "market"}
    )
    fitted_columns = [f"fitted_probability_{CATEGORY_SHORT[c]}" for c in CATEGORIES]
    fitted = row_output.groupby("frame_id", as_index=False, observed=True)[fitted_columns].mean()
    effects = row_output.groupby("frame_id", as_index=False, observed=True)[effect_columns].first()
    weights = metadata.merge(fitted, on="frame_id", validate="one_to_one").merge(
        effects, on="frame_id", validate="one_to_one"
    )
    weights["fitted_probability_sum"] = weights[fitted_columns].sum(axis=1)
    weights["dominant_fitted_category"] = weights[fitted_columns].idxmax(axis=1).str.replace(
        "fitted_probability_", "", regex=False
    )
    return row_output, weights


def regression_design(weights: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "const": 1.0,
        "b_M_vs_N": weights["b_M_vs_N"].astype(float),
        "b_S_vs_N": weights["b_S_vs_N"].astype(float),
        "b_C_vs_N": weights["b_C_vs_N"].astype(float),
        "cell_lt_control": (weights["cell"] == "lt_control").astype(float),
        "cell_kw_market": (weights["cell"] == "kw_market").astype(float),
        "cell_lt_market": (weights["cell"] == "lt_market").astype(float),
    }, index=weights.index)


def fit_ols_clustered(
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    design = regression_design(weights)
    result = sm.OLS(weights["share_sent"].astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": weights["PROLIFIC_PID"], "use_correction": True},
    )
    confidence = result.conf_int(alpha=0.05)
    rows = []
    for term in result.params.index:
        rows.append({
            "model": "partial_pooling_cell_adjusted",
            "outcome": "share_sent",
            "term": term,
            "coefficient": result.params[term],
            "std_error_clustered_pid": result.bse[term],
            "ci95_low": confidence.loc[term, 0],
            "ci95_high": confidence.loc[term, 1],
            "p_value": result.pvalues[term],
            "n_frames": int(result.nobs),
            "n_unique_pid": weights["PROLIFIC_PID"].nunique(),
            "r_squared": result.rsquared,
            "outcome_mean": weights["share_sent"].mean(),
            "first_stage_reference_category": "N (No clear justification)",
            "cell_reference": "kw_control",
            "predictor_std_dev": float(design[term].std(ddof=1)),
        })
    return pd.DataFrame(rows), result


def grouped_behavior_cv(weights: pd.DataFrame, n_folds: int = 10) -> pd.DataFrame:
    pids = np.array(sorted(weights["PROLIFIC_PID"].unique()))
    rng = np.random.default_rng(20260824)
    rng.shuffle(pids)
    pid_fold = {pid: index % n_folds for index, pid in enumerate(pids)}
    fold = weights["PROLIFIC_PID"].map(pid_fold).to_numpy(dtype=int)
    outcome = weights["share_sent"].to_numpy(dtype=float)
    full_design = regression_design(weights).to_numpy(dtype=float)
    baseline_design = full_design[:, [0, 4, 5, 6]]
    predictions = {
        "cell_only": np.full(len(weights), np.nan),
        "cell_plus_participant_effects": np.full(len(weights), np.nan),
    }
    rows = []
    for held_fold in range(n_folds):
        train = fold != held_fold
        test = ~train
        for model_name, design in [
            ("cell_only", baseline_design),
            ("cell_plus_participant_effects", full_design),
        ]:
            coefficients = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            prediction = design[test] @ coefficients
            predictions[model_name][test] = prediction
            rows.append({
                "scope": "fold",
                "fold": held_fold + 1,
                "model": model_name,
                "n_test_frames": int(test.sum()),
                "rmse": float(np.sqrt(np.mean((outcome[test] - prediction) ** 2))),
                "mae": float(np.mean(np.abs(outcome[test] - prediction))),
            })
    total_variation = float(np.sum((outcome - outcome.mean()) ** 2))
    for model_name, prediction in predictions.items():
        squared_error = float(np.sum((outcome - prediction) ** 2))
        rows.append({
            "scope": "pooled",
            "fold": np.nan,
            "model": model_name,
            "n_test_frames": len(weights),
            "rmse": float(np.sqrt(np.mean((outcome - prediction) ** 2))),
            "mae": float(np.mean(np.abs(outcome - prediction))),
            "out_of_sample_r_squared": 1.0 - squared_error / total_variation,
        })
    output = pd.DataFrame(rows)
    pooled = output[output["scope"] == "pooled"].set_index("model")
    improvement = {
        "scope": "increment",
        "fold": np.nan,
        "model": "participant_effects_minus_cell_only",
        "n_test_frames": len(weights),
        "rmse": pooled.loc["cell_plus_participant_effects", "rmse"]
        - pooled.loc["cell_only", "rmse"],
        "mae": pooled.loc["cell_plus_participant_effects", "mae"]
        - pooled.loc["cell_only", "mae"],
        "out_of_sample_r_squared": pooled.loc[
            "cell_plus_participant_effects", "out_of_sample_r_squared"
        ] - pooled.loc["cell_only", "out_of_sample_r_squared"],
    }
    return pd.concat([output, pd.DataFrame([improvement])], ignore_index=True)


def summary_text(
    data: pd.DataFrame,
    weights: pd.DataFrame,
    selected_lambda: float,
    fit: FirstStageFit,
    regressions: pd.DataFrame,
    behavior_cv: pd.DataFrame,
    source_hash: str,
    fingerprint: str,
) -> str:
    category_counts = data["category"].value_counts().reindex(CATEGORIES)
    repeats = weights.groupby("PROLIFIC_PID", observed=True).size()
    main = regressions[
        (regressions["model"] == "partial_pooling_cell_adjusted")
        & (regressions["outcome"] == "share_sent")
        & regressions["term"].isin(["b_M_vs_N", "b_S_vs_N", "b_C_vs_N"])
    ]
    pooled_cv = behavior_cv[behavior_cv["scope"].isin(["pooled", "increment"])]
    lines = [
        "Participant-level HP effect exercise (four categories)",
        "=" * 62,
        f"Input: {INPUT.name}",
        f"Input SHA-256: {source_hash}",
        f"Model fingerprint: {fingerprint}",
        "",
        "Sample and unit",
        f"- {len(data):,} HP responses.",
        f"- {len(weights):,} participant-game-condition frames.",
        f"- {weights['PROLIFIC_PID'].nunique():,} unique Prolific IDs.",
        f"- {int((repeats > 1).sum())} Prolific IDs appear in more than one frame.",
        f"- Frame cells: {weights['cell'].value_counts().sort_index().to_dict()}.",
        f"- Category counts: {category_counts.to_dict()}.",
        "- Every frame contributes classifications at HP=4, 6, 8, and 12.",
        "",
        "First stage",
        "- Multinomial logit; No clear justification (N) is the reference.",
        "- HP effects are unpenalized; participant-frame category effects are L2-shrunk.",
        "- Lambda chosen by four-fold CV holding out one HP response per frame per fold.",
        f"- Selected lambda: {selected_lambda:g}.",
        f"- Optimizer converged: {fit.success}; iterations: {fit.iterations}; "
        f"max |gradient|: {fit.gradient_max:.3g}; objective: {fit.objective:.3f}.",
        f"- Optimizer message: {fit.message}",
        "- The direct participant-frame effects b_M, b_S, and b_C are retained for the second stage.",
        "",
        "Second stage",
        "- Main outcome: share sent = (12 - allocation kept)/12.",
        "- Predictors: b_M, b_S, and b_C, each measured relative to N in first-stage log odds.",
        "- Controls: indicators for LT Control, KW Market, and LT Market; reference cell is KW Control.",
        "- Standard errors clustered by Prolific ID.",
        "- Coefficients are changes in share sent for a one-unit increase in the indicated log-odds effect.",
        "",
        "Main cell-adjusted coefficients (share sent)",
    ]
    for row in main.itertuples(index=False):
        lines.append(
            f"- {row.term}: {row.coefficient:+.4f} "
            f"(clustered SE {row.std_error_clustered_pid:.4f}, "
            f"95% CI [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}], p={row.p_value:.4g})."
        )
    lines.extend(["", "Grouped 10-fold behavioral prediction", pooled_cv.to_string(index=False), ""])
    return "\n".join(lines)


def run_fit(lambda_grid: list[float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_and_validate()
    source_hash = file_sha256(INPUT)
    fingerprint = model_fingerprint(source_hash, lambda_grid)
    existing = read_progress()
    bootstrap_path = OUT / "bootstrap_draws.csv"
    if bootstrap_path.exists() and existing.get("model_fingerprint") not in [None, fingerprint]:
        raise RuntimeError(
            "Existing bootstrap draws belong to a different model. Archive the output folder first."
        )
    write_progress(
        stage="tuning_first_stage",
        source_sha256=source_hash,
        model_fingerprint=fingerprint,
        lambda_grid=lambda_grid,
        bootstrap_completed=int(existing.get("bootstrap_completed", 0)),
    )
    lambda_cv = tune_penalty(data, lambda_grid)
    selected_lambda = float(
        lambda_cv.groupby("penalty_lambda", observed=True)["log_loss"].mean().idxmin()
    )
    atomic_write_csv(lambda_cv, OUT / "first_stage_lambda_cv.csv")

    write_progress(stage="fitting_first_stage", selected_lambda=selected_lambda)
    fixed_design, group_codes, outcome, fixed_names, _ = prepare_first_stage(data)
    fit = fit_penalized_multinomial(fixed_design, group_codes, outcome, selected_lambda)
    if not fit.success and fit.gradient_max > 1e-4:
        raise RuntimeError(
            f"First-stage optimization failed: {fit.message}; "
            f"max |gradient|={fit.gradient_max:.3g}."
        )
    atomic_write_csv(fixed_effect_table(fit, fixed_names), OUT / "first_stage_fixed_effects.csv")
    fitted_rows, weights = build_fitted_outputs(data, fit, fixed_design, group_codes)
    if not np.allclose(weights["fitted_probability_sum"], 1.0, atol=1e-10):
        raise AssertionError("Participant-frame fitted probabilities do not sum to one.")
    atomic_write_csv(fitted_rows, OUT / "hp_fitted_probabilities.csv")
    atomic_write_csv(weights, OUT / "participant_frame_effects.csv")

    write_progress(stage="fitting_second_stage")
    regressions, _ = fit_ols_clustered(weights)
    behavior_cv = grouped_behavior_cv(weights)
    atomic_write_csv(regressions, OUT / "second_stage_regressions.csv")
    atomic_write_csv(behavior_cv, OUT / "second_stage_grouped_cv.csv")
    atomic_write_text(
        OUT / "analysis_summary.txt",
        summary_text(
            data, weights, selected_lambda, fit, regressions, behavior_cv,
            source_hash, fingerprint,
        ),
    )
    write_progress(
        stage="fit_complete",
        fit_complete=True,
        selected_lambda=selected_lambda,
        final_converged=fit.success,
        final_iterations=fit.iterations,
        final_gradient_max=fit.gradient_max,
        n_hp_rows=len(data),
        n_frames=len(weights),
        n_unique_pid=weights["PROLIFIC_PID"].nunique(),
    )


def bootstrap_sample(data: pd.DataFrame, replicate: int) -> pd.DataFrame:
    unique_pids = np.array(sorted(data["PROLIFIC_PID"].unique()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + replicate)
    sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
    cluster_draws = pd.DataFrame({
        "_original_pid": sampled,
        "bootstrap_cluster": [f"b{index:04d}" for index in range(len(sampled))],
    })
    sample = cluster_draws.merge(
        data,
        left_on="_original_pid",
        right_on="PROLIFIC_PID",
        how="left",
        sort=False,
        validate="many_to_many",
    )
    sample["frame_id"] = sample["bootstrap_cluster"] + "|" + sample["frame_id"]
    sample["PROLIFIC_PID"] = sample["bootstrap_cluster"]
    return sample.drop(columns="_original_pid").reset_index(drop=True)


def bootstrap_one(data: pd.DataFrame, penalty: float, replicate: int) -> list[dict]:
    sample = bootstrap_sample(data, replicate)
    fixed_design, group_codes, outcome, _, _ = prepare_first_stage(sample)
    fit = fit_penalized_multinomial(
        fixed_design, group_codes, outcome, penalty, maxiter=350
    )
    _, weights = build_fitted_outputs(sample, fit, fixed_design, group_codes)
    _, result = fit_ols_clustered(weights)
    rows = []
    for term in ["b_M_vs_N", "b_S_vs_N", "b_C_vs_N"]:
        rows.append({
            "replicate": replicate,
            "term": term,
            "coefficient": float(result.params[term]),
            "first_stage_converged": fit.success,
            "first_stage_iterations": fit.iterations,
            "first_stage_gradient_max": fit.gradient_max,
            "n_frames": len(weights),
            "n_bootstrap_clusters": weights["PROLIFIC_PID"].nunique(),
        })
    return rows


def summarize_bootstrap(draws: pd.DataFrame, regressions: pd.DataFrame) -> pd.DataFrame:
    point = regressions[
        (regressions["model"] == "partial_pooling_cell_adjusted")
        & (regressions["outcome"] == "share_sent")
        & regressions["term"].isin(["b_M_vs_N", "b_S_vs_N", "b_C_vs_N"])
    ].set_index("term")["coefficient"]
    records = []
    for term, group in draws.groupby("term", observed=True):
        records.append({
            "term": term,
            "point_estimate": point[term],
            "bootstrap_standard_error": group["coefficient"].std(ddof=1),
            "bootstrap_ci95_low_percentile": group["coefficient"].quantile(0.025),
            "bootstrap_ci95_high_percentile": group["coefficient"].quantile(0.975),
            "bootstrap_replicates": group["replicate"].nunique(),
            "failed_first_stage_fits": int((~group["first_stage_converged"]).sum()),
        })
    return pd.DataFrame(records)


def add_bootstrap_to_summary(summary: pd.DataFrame) -> None:
    path = OUT / "analysis_summary.txt"
    text = path.read_text(encoding="utf-8")
    marker = "Whole-pipeline participant-cluster bootstrap"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    lines = [
        "", "", marker,
        "- Both stages are refitted in every replicate; the shrinkage penalty is held at "
        "the full-sample cross-validated value.",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.term}: point estimate {row.point_estimate:+.4f}; "
            f"bootstrap SE {row.bootstrap_standard_error:.4f}; percentile 95% CI "
            f"[{row.bootstrap_ci95_low_percentile:+.4f}, "
            f"{row.bootstrap_ci95_high_percentile:+.4f}] "
            f"({int(row.bootstrap_replicates)} replicates; "
            f"{int(row.failed_first_stage_fits)} failed first-stage fits)."
        )
    atomic_write_text(path, text + "\n".join(lines) + "\n")


def moral_baseline_tex_section() -> list[str]:
    moral_out = ROOT / "output" / "moral_baseline_participant_effects"
    required = [
        moral_out / "second_stage_regressions.csv",
        moral_out / "bootstrap_summary.csv",
        moral_out / "second_stage_grouped_cv.csv",
        moral_out / "sample_summary.json",
    ]
    if not all(path.exists() for path in required):
        return []
    regression = pd.read_csv(required[0]).set_index("term")
    bootstrap = pd.read_csv(required[1]).set_index("term")
    behavior_cv = pd.read_csv(required[2])
    sample = json.loads(required[3].read_text(encoding="utf-8"))
    pooled_cv = behavior_cv[behavior_cv["scope"] == "pooled"].set_index("model")
    labels = {
        "b_S_vs_M": r"$b_{Si}$: Self-interest vs. Moral",
        "b_C_vs_M": r"$b_{Ci}$: Cooperation vs. Moral",
    }
    rows = []
    for term in ["b_S_vs_M", "b_C_vs_M"]:
        row = regression.loc[term]
        boot = bootstrap.loc[term]
        p_value = (
            r"$<0.001$"
            if row["p_value"] < 0.001
            else f"{row['p_value']:.3f}"
        )
        rows.append(
            f"{labels[term]} & {row['coefficient']:.3f} & "
            f"{row['std_error_clustered_pid']:.3f} & {p_value} & "
            f"[{boot['bootstrap_ci95_low_percentile']:.3f}, "
            f"{boot['bootstrap_ci95_high_percentile']:.3f}] & "
            f"{row['coefficient'] * row['predictor_std_dev']:+.3f} \\\\"
        )
    cell_only = pooled_cv.loc["cell_only"]
    full = pooled_cv.loc["cell_plus_participant_effects"]
    incremental_r2 = full["out_of_sample_r_squared"] - cell_only["out_of_sample_r_squared"]
    s_boot = bootstrap.loc["b_S_vs_M"]
    c_boot = bootstrap.loc["b_C_vs_M"]
    s_result = (
        "excludes zero"
        if s_boot["bootstrap_ci95_low_percentile"] > 0
        or s_boot["bootstrap_ci95_high_percentile"] < 0
        else "includes zero"
    )
    c_result = (
        "excludes zero"
        if c_boot["bootstrap_ci95_low_percentile"] > 0
        or c_boot["bootstrap_ci95_high_percentile"] < 0
        else "includes zero"
    )
    return [
        "",
        r"\subsection{Moral baseline, excluding No clear justification}",
        "",
        r"We remove all No clear justification classifications and re-estimate the "
        r"model on M, S, and C, using Moral as the reference category:",
        "",
        r"\begin{equation}",
        r"    \log\frac{\Pr(K_{ia}=k)}{\Pr(K_{ia}=M)}=\alpha_{ka}+b_{ki},",
        r"    \qquad k\in\{S,C\}.",
        r"\end{equation}",
        "",
        f"The analysis retains {int(sample['n_substantive_hp_rows']):,} substantive HP "
        f"classifications from {int(sample['n_usable_frames']):,} participant-frames. "
        f"The {int(sample['n_excluded_all_no_clear_frames'])} frames containing only "
        r"No clear classifications are excluded. The estimated $b_{Si}$ and "
        r"$b_{Ci}$ are participant-frame deviations in the log odds of Self-interest "
        r"and Cooperation relative to Moral, net of differences across the four "
        r"hypothetical allocations.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Moral-baseline participant effects and DG share sent}",
        r"\label{tab:hp_weights_moral_baseline}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Participant effect & Coefficient & Clustered SE & $p$-value & Bootstrap 95\% CI "
        r"& Effect of +1 SD \\",
        r"\midrule",
        *rows,
        r"\midrule",
        f"Observations & {int(regression.loc['const', 'n_frames'])} & & & & \\\\",
        f"$R^2$ & {regression.loc['const', 'r_squared']:.3f} & & & & \\\\",
        r"Four game-by-condition cells & Yes & & & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Moral is the first-stage reference category. Coefficients "
        r"give the change in share sent for a one-unit increase in the indicated "
        r"participant log-odds effect. The final column rescales this change to one "
        r"sample standard deviation. Confidence intervals use 500 bootstrap samples "
        r"drawn at the participant level.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        f"The bootstrap interval for the Self-interest participant effect {s_result}; "
        f"the interval for the Cooperation participant effect "
        f"{c_result}. In grouped 10-fold cross-validation, adding the participant "
        f"effects changes out-of-sample $R^2$ from "
        f"{cell_only['out_of_sample_r_squared']:.3f} to "
        f"{full['out_of_sample_r_squared']:.3f} "
        f"($\\Delta R^2={incremental_r2:.3f}$).",
    ]


def heterogeneity_tex_section() -> list[str]:
    heterogeneity_out = ROOT / "output" / "heterogeneity_participant_effects"
    required = [
        heterogeneity_out / "point_estimates.csv",
        heterogeneity_out / "bootstrap_summary.csv",
        heterogeneity_out / "joint_tests.csv",
    ]
    if not all(path.exists() for path in required):
        return []
    point = pd.read_csv(required[0])
    bootstrap = pd.read_csv(required[1]).set_index(
        ["dimension", "category_vs_moral", "estimand"]
    )
    joint = pd.read_csv(required[2]).set_index("dimension")
    labels = {
        "S": r"$b_{Si}$: Self-interest vs. Moral",
        "C": r"$b_{Ci}$: Cooperation vs. Moral",
    }
    panels = []
    panel_specs = [
        ("game", "Panel A: By DG game", "KW", "LT", "LT $-$ KW"),
        (
            "condition",
            "Panel B: By condition, KW only",
            "KW Control",
            "KW Market",
            "Market $-$ Control",
        ),
    ]
    for (
        dimension,
        panel_title,
        base_label,
        comparison_label,
        difference_label,
    ) in panel_specs:
        panels.extend([
            rf"\multicolumn{{6}}{{l}}{{\textit{{{panel_title}}}}} \\",
            (
            rf"Participant effect & {base_label} slope & {comparison_label} slope & "
                rf"{difference_label} & Bootstrap 95\% CI & $p$-value \\"
            ),
        ])
        subset = point[point["dimension"] == dimension].set_index(
            "category_vs_moral"
        )
        for category in ["S", "C"]:
            row = subset.loc[category]
            boot = bootstrap.loc[(dimension, category, "difference")]
            p_value = (
                r"$<0.001$"
                if row["difference_p_value"] < 0.001
                else f"{row['difference_p_value']:.3f}"
            )
            panels.append(
                f"{labels[category]} & {row['base_slope_estimate']:.3f} "
                f"({row['base_slope_std_error']:.3f}) & "
                f"{row['comparison_slope_estimate']:.3f} "
                f"({row['comparison_slope_std_error']:.3f}) & "
                f"{row['difference_estimate']:.3f} "
                f"({row['difference_std_error']:.3f}) & "
                f"[{boot['bootstrap_ci95_low_percentile']:.3f}, "
                f"{boot['bootstrap_ci95_high_percentile']:.3f}] & "
                f"{p_value} \\\\"
            )
        panels.extend([
            (
                r"\multicolumn{5}{l}{Joint test of both interactions} & "
                f"{joint.loc[dimension, 'p_value']:.3f} \\\\"
            ),
            r"\addlinespace",
        ])
    game_joint = joint.loc["game", "p_value"]
    condition_joint = joint.loc["condition", "p_value"]
    game_conclusion = (
        "evidence of" if game_joint < 0.05 else "no clear evidence of"
    )
    condition_conclusion = (
        "evidence of" if condition_joint < 0.05 else "no clear evidence of"
    )
    return [
        "",
        r"\subsection{Heterogeneity by game and condition}",
        "",
        r"Using the Moral-baseline sample, we interact $b_{Si}$ and $b_{Ci}$ first "
        r"with an LT indicator, using KW as the reference game. This regression "
        r"includes indicators for LT Control, KW Market, and LT Market, with KW "
        r"Control omitted; hence average allocations may differ across all four "
        r"game-by-condition cells. For the condition comparison, we retain only KW "
        r"observations and interact the participant effects with a Market indicator, "
        r"using KW Control as the reference cell.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Heterogeneity in participant-effect slopes}",
        r"\label{tab:hp_weight_heterogeneity}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        *panels,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Moral is the first-stage reference category and "
        r"classifications labelled No clear justification are excluded. Slopes refer "
        r"to the direct participant log-odds "
        r"effects. Entries in parentheses are participant-"
        r"clustered standard errors. Bootstrap intervals are for the slope "
        r"difference and refit both stages in 500 bootstrap samples drawn at the "
        r"participant level.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        f"The joint tests provide {game_conclusion} heterogeneity by game "
        f"($p={game_joint:.3f}$) and {condition_conclusion} heterogeneity by "
        f"condition ($p={condition_joint:.3f}$).",
    ]


def within_subject_tex_section_legacy() -> list[str]:
    within_out = ROOT / "within_subject" / "output"
    required = [
        within_out / "sample_summary.json",
        within_out / "first_stage_treatment_cell_probabilities.csv",
        within_out / "allocation_component_tests.csv",
        within_out / "allocation_models.csv",
        within_out / "reasoning_models.csv",
        within_out / "reasoning_component_tests.csv",
        within_out / "reasoning_model_performance.csv",
    ]
    if not all(path.exists() for path in required):
        return []

    sample = json.loads(required[0].read_text(encoding="utf-8"))
    cell = pd.read_csv(required[1]).set_index("treatment_cell")
    allocation_tests = pd.read_csv(required[2]).set_index(
        ["period", "component"]
    )
    allocation = pd.read_csv(required[3])
    reasoning = pd.read_csv(required[4])
    reasoning_tests = pd.read_csv(required[5]).set_index(
        ["period", "component"]
    )
    reasoning_performance = pd.read_csv(required[6]).set_index(
        ["period", "model"]
    )

    def p_tex(value: float) -> str:
        return r"$<0.001$" if value < 0.001 else f"{value:.3f}"

    def p_inline(value: float) -> str:
        return r"$p<0.001$" if value < 0.001 else f"$p={value:.3f}$"

    cell_labels = {
        "lt_control": "LT Control",
        "kw_control": "KW Control",
        "lt_market": "LT Market",
        "kw_market": "KW Market",
    }
    cell_rows = []
    for key in ["lt_control", "kw_control", "lt_market", "kw_market"]:
        row = cell.loc[key]
        cell_rows.append(
            f"{cell_labels[key]} & {row['probability_M']:.3f} & "
            f"{row['probability_S']:.3f} & {row['probability_C']:.3f} \\\\"
        )

    predictor_labels = {
        "subject_weight_S": "Participant S weight",
        "subject_weight_C": "Participant C weight",
        "treatment_shift_S": "Treatment-induced S shift",
        "treatment_shift_C": "Treatment-induced C shift",
    }
    predictor_order = list(predictor_labels)
    allocation_rows = []
    reasoning_rows = []
    for period in [1, 2]:
        subject = allocation_tests.loc[(period, "subject")]
        treatment = allocation_tests.loc[(period, "treatment")]
        full_allocation = allocation[
            (allocation["period"] == period) & (allocation["model"] == "full")
        ].set_index("term")
        panel = "First played game" if period == 1 else "Second played game"
        allocation_rows.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{{panel}}}}} \\"
        )
        for term in predictor_order:
            row = full_allocation.loc[term]
            allocation_rows.append(
                f"{predictor_labels[term]} & {row['coefficient']:.3f} & "
                f"{row['std_error_hc3']:.3f} & {p_tex(row['p_value'])} & "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] & "
                f"{10.0 * row['coefficient']:+.2f} \\\\"
            )
        allocation_rows.extend([
            rf"\multicolumn{{6}}{{l}}{{N={int(full_allocation.iloc[0]['n'])}; "
            rf"$R^2={full_allocation.iloc[0]['r_squared']:.3f}$; joint participant "
            f"{p_inline(subject['p_value'])}; joint treatment "
            f"{p_inline(treatment['p_value'])}.}} \\\\",
            r"\addlinespace",
        ])

        subject_r = reasoning_tests.loc[(period, "subject")]
        treatment_r = reasoning_tests.loc[(period, "treatment")]
        full_reasoning = reasoning_performance.loc[(period, "full")]
        full_reasoning_coefficient = reasoning[
            reasoning["period"] == period
        ].set_index(["term", "outcome_vs_moral"])
        reasoning_rows.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{panel}}}}} \\"
        )
        for term in predictor_order:
            s_row = full_reasoning_coefficient.loc[(term, "S")]
            c_row = full_reasoning_coefficient.loc[(term, "C")]
            reasoning_rows.append(
                f"{predictor_labels[term]} & "
                f"{s_row['coefficient']:.3f} ({s_row['std_error']:.3f}) & "
                f"{p_tex(s_row['p_value'])} & "
                f"{c_row['coefficient']:.3f} ({c_row['std_error']:.3f}) & "
                f"{p_tex(c_row['p_value'])} \\\\"
            )
        reasoning_rows.extend([
            rf"\multicolumn{{5}}{{l}}{{N={int(full_reasoning['n'])}; pseudo-$R^2="
            rf"{full_reasoning['pseudo_r_squared']:.3f}$; joint participant "
            f"{p_inline(subject_r['p_value'])}; joint treatment "
            f"{p_inline(treatment_r['p_value'])}.}} \\\\",
            r"\addlinespace",
        ])

    allocation_subject_text = (
        r"The participant HP components do not jointly predict allocation in either "
        r"period."
        if all(
            allocation_tests.loc[(period, "subject"), "p_value"] >= 0.05
            for period in [1, 2]
        )
        else r"The participant HP components jointly predict allocation in at least "
        r"one period."
    )
    reasoning_subject_p = {
        period: reasoning_tests.loc[(period, "subject"), "p_value"]
        for period in [1, 2]
    }
    if reasoning_subject_p[1] >= 0.05 and reasoning_subject_p[2] < 0.05:
        reasoning_subject_text = (
            r"The participant HP components do not jointly predict stated reasoning "
            r"in the first game, but they do in the second."
        )
    elif all(value < 0.05 for value in reasoning_subject_p.values()):
        reasoning_subject_text = (
            r"The participant HP components jointly predict stated reasoning in both "
            r"periods."
        )
    else:
        reasoning_subject_text = (
            r"The participant HP components do not jointly predict stated reasoning "
            r"in either period."
        )

    return [
        "",
        r"\section{Within-subject analysis}",
        "",
        r"\subsection{Classification and first stage}",
        "",
        f"The within-subject data contain {int(sample['n_subjects_all']):,} participants "
        f"and {int(sample['n_hp_rows_all']):,} HP responses from two pre-choice "
        r"elicitations. HP descriptions are classified with the same M/S/C/N "
        r"prompt as in the between-subject analysis. We exclude N responses and estimate",
        "",
        r"\begin{equation}",
        r" \log\frac{\Pr(K_{iat}=k)}{\Pr(K_{iat}=M)}="
        r"\alpha_{ka}+\tau_{k1}\mathrm{Market}_{it}+\tau_{k2}\mathrm{KW}_{it}"
        r"+\tau_{k3}(\mathrm{Market}\times\mathrm{KW})_{it}+b_{ki},"
        r" \qquad k\in\{S,C\}.",
        r"\end{equation}",
        "",
        f"The estimation sample contains {int(sample['n_hp_rows_substantive']):,} "
        f"substantive classifications and {int(sample['n_subjects_usable']):,} "
        r"participants; participants whose HP descriptions are all N are excluded. "
        r"Allocation, treatment, and regularized participant effects are estimated "
        rf"jointly; cross-validation selects $\lambda={sample['selected_lambda']:.3g}$.",
        r"In the current within-subject specification, we express the participant effects as "
        r"M/S/C prevalence weights. For participant $i$, $w^P_{ik}$ is the fitted "
        r"probability of category $k$, averaged over the four allocation anchors and "
        r"evaluated at the common LT-Control reference. The treatment component "
        r"$\Delta^T_{itk}$ is the probability shift from LT Control to the participant's "
        r"current Market/KW cell, evaluated at zero participant effect. Both sets of "
        r"components sum to one (weights) or zero (shifts), so Moral is omitted.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{First-stage fitted category probabilities by treatment cell}",
        r"\label{tab:within_first_stage_cells}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Cell & Moral & Self-interest & Cooperation \\",
        r"\midrule",
        *cell_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Probabilities are averaged over the four allocation "
        r"anchors and evaluated at zero participant effect. They are conditional on an "
        r"M/S/C classification.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"\subsection{Predicting allocations}",
        "",
        r"We predict share sent separately in the first and second played game. The "
        r"full model contains the participant S and C weights, the treatment-induced S "
        r"and C probability shifts, and an indicator for the source experiment. HC3 standard errors "
        r"are used.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Within-subject HP components and allocation}",
        r"\label{tab:within_allocation_prediction}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Component & Coefficient & HC3 SE & $p$-value & 95\% CI & Effect of +10 pp \\",
        r"\midrule",
        *allocation_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Moral is omitted. Coefficients correspond to a "
        r"100-percentage-point change in the indicated probability component. The last "
        r"column gives the percentage-point change in share sent associated with a "
        r"10-percentage-point change. Joint tests cover the S and C terms within each "
        r"component. Standard errors and confidence intervals condition on the "
        r"first-stage generated components.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        allocation_subject_text + " Treatment components are strongly predictive in "
        r"both periods, with larger incremental explanatory power in the first game.",
        "",
        r"\subsection{Predicting post-choice reasoning}",
        "",
        r"We next estimate period-specific multinomial logits for the classified "
        r"post-choice reason (Moral omitted), excluding N outcomes.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Within-subject HP components and post-choice reasoning}",
        r"\label{tab:within_reasoning_prediction}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Component & S vs M coefficient (SE) & $p$-value & "
        r"C vs M coefficient (SE) & $p$-value \\",
        r"\midrule",
        *reasoning_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: The outcome is the classified post-choice reason; Moral "
        r"is the omitted outcome. Coefficients are multinomial-logit coefficients for a "
        r"100-percentage-point change in the indicated component. Likelihood-ratio "
        r"tests jointly test the four category-by-component coefficients. Inference "
        r"conditions on the first-stage generated components.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        reasoning_subject_text + " Treatment components predict stated reasoning in "
        r"both periods, although their incremental fit is substantially larger for "
        r"the first game than for the second.",
        "",
        r"All within-subject results are descriptive, in-sample associations. In "
        r"particular, $w^P_{ik}$ pools both HP elicitations, so the first-game exercise is "
        r"not a prospective prediction based only on information observed before the "
        r"first game.",
    ]


def within_subject_tex_section() -> list[str]:
    """Render the final cell-specific within-subject specification."""
    within_out = ROOT / "within_subject" / "output" / "cell_specific"
    names = [
        "sample_summary.json",
        "cell_weight_means.csv",
        "pooled_allocation_model.csv",
        "pooled_reasoning_model.csv",
        "pooled_reasoning_performance.csv",
        "treatment_shift_decomposition.csv",
        "moral_weight_transition_matrices.csv",
        "moral_weight_tercile_cutoffs.csv",
    ]
    paths = [within_out / name for name in names]
    if not all(path.exists() for path in paths):
        return []

    sample = json.loads(paths[0].read_text(encoding="utf-8"))
    means = pd.read_csv(paths[1]).set_index("treatment_cell")
    allocation = pd.read_csv(paths[2]).set_index("term")
    reasoning = pd.read_csv(paths[3]).set_index(["term", "outcome_vs_moral"])
    reasoning_fit = pd.read_csv(paths[4]).iloc[0]
    shifts = pd.read_csv(paths[5]).set_index("comparison")
    transitions = pd.read_csv(paths[6])
    cutoffs = pd.read_csv(paths[7]).iloc[0]

    def p_tex(value: float) -> str:
        return r"$<0.001$" if value < 0.001 else f"{value:.3f}"

    cell_labels = {
        "kw_control": "KW Control",
        "lt_control": "LT Control",
        "kw_market": "KW Market",
        "lt_market": "LT Market",
    }
    cell_summary = {item["treatment_cell"]: item for item in sample["cells"]}
    cell_rows = []
    for key in cell_labels:
        row = means.loc[key]
        detail = cell_summary[key]
        cell_rows.append(
            f"{cell_labels[key]} & {row['weight_M']:.3f} & "
            f"{row['weight_S']:.3f} & {row['weight_C']:.3f} & "
            f"{int(detail['n_elicitations_usable'])} & "
            f"{detail['selected_lambda']:.3g} \\\\"
        )

    allocation_rows = []
    for term, label in [
        ("weight_S", "S weight (M omitted)"),
        ("weight_C", "C weight (M omitted)"),
    ]:
        row = allocation.loc[term]
        allocation_rows.append(
            f"{label} & {row['coefficient']:.3f} & "
            f"{row['std_error_clustered_pid']:.3f} & "
            f"{p_tex(row['p_value'])} & "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] & "
            f"{10 * row['coefficient']:+.2f} \\\\"
        )

    reasoning_rows = []
    for term, label in [("weight_S", "S weight"), ("weight_C", "C weight")]:
        s_row = reasoning.loc[(term, "S")]
        c_row = reasoning.loc[(term, "C")]
        reasoning_rows.append(
            f"{label} & {s_row['coefficient']:.3f} "
            f"({s_row['std_error_clustered_pid']:.3f}) & "
            f"{p_tex(s_row['p_value'])} & {c_row['coefficient']:.3f} "
            f"({c_row['std_error_clustered_pid']:.3f}) & "
            f"{p_tex(c_row['p_value'])} \\\\"
        )

    comparison_labels = {
        "KW Control to LT Control": r"KW Control $\rightarrow$ LT Control",
        "KW Control to KW Market": r"KW Control $\rightarrow$ KW Market",
    }
    shift_rows = []
    for key, label in comparison_labels.items():
        row = shifts.loc[key]
        shift_rows.append(
            f"{label} & {100 * row['mean_delta_weight_M']:+.1f} & "
            f"{100 * row['mean_delta_weight_S']:+.1f} & "
            f"{100 * row['mean_delta_weight_C']:+.1f} & "
            f"{100 * row['predicted_allocation_shift_from_weights']:+.1f} & "
            f"{100 * row['actual_mean_allocation_shift']:+.1f} & "
            f"{int(row['n_paired_participants'])} \\\\"
        )

    transition_rows = []
    for comparison, label in comparison_labels.items():
        transition_rows.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{label}}}}} \\"
        )
        subset = transitions[transitions["comparison"] == comparison]
        for origin in ["Low", "Middle", "High"]:
            row = subset[subset["origin_tercile"] == origin].set_index(
                "destination_tercile"
            )
            entries = []
            for destination in ["Low", "Middle", "High"]:
                item = row.loc[destination]
                entries.append(f"{item['row_percent']:.1f} ({int(item['count'])})")
            transition_rows.append(
                f"{origin} & {entries[0]} & {entries[1]} & {entries[2]} & "
                f"{int(row.iloc[0]['row_n'])} \\\\"
            )
        transition_rows.append(r"\addlinespace")

    return [
        "",
        r"\section{Within-subject analysis}",
        "",
        r"\subsection{Cell-specific HP weights}",
        "",
        f"The within-subject data contain {int(sample['n_subjects_all']):,} participants "
        f"and {int(sample['n_hp_rows_all']):,} HP responses from two pre-choice "
        r"elicitations. We apply the same M/S/C/N classification prompt, exclude "
        r"N responses, and estimate separately for each game-by-condition cell $c$:",
        "",
        r"\begin{equation}",
        r"\log\frac{\Pr(K_{ia}=k\mid c)}{\Pr(K_{ia}=M\mid c)}="
        r"\alpha^c_{ka}+b^c_{ki},\qquad k\in\{S,C\}.",
        r"\end{equation}",
        "",
        f"The models use {int(sample['n_hp_rows_substantive']):,} HP responses "
        f"classified as M, S, or C and retain "
        f"{int(sample['n_usable_participant_cells']):,} "
        r"participant-by-treatment-cell observations. An observation is excluded when "
        r"all four HP responses are classified N. Participant effects are regularized "
        r"toward zero, with the strength selected separately by within-cell "
        r"cross-validation. M/S/C weights are the fitted "
        r"probabilities averaged over the four allocation anchors.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Average cell-specific HP weights}",
        r"\label{tab:within_first_stage_cells}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Cell & Moral & Self-interest & Cooperation & Observations & $\lambda$ \\",
        r"\midrule",
        *cell_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: N classifications are excluded, so the M/S/C weights sum "
        r"to one. The table averages each participant's fitted probabilities over the "
        r"four allocation anchors and then across participants in the cell.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"\subsection{Pooled prediction of allocation and reasoning}",
        "",
        r"We pool first- and second-game observations and regress share sent on S and C "
        r"weights, omitting Moral. The model includes game-by-condition cell fixed "
        r"effects, a second-game indicator, and an indicator for the source experiment. Standard "
        r"errors are clustered by participant.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cell-specific HP weights and share sent}",
        r"\label{tab:within_allocation_prediction}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Weight & Coefficient & Clustered SE & $p$-value & 95\% CI & Effect of +10 pp \\",
        r"\midrule",
        *allocation_rows,
        r"\midrule",
        f"Observations & {int(allocation.loc['const', 'n'])} & & & & \\\\ ",
        f"$R^2$ & {allocation.loc['const', 'r_squared']:.3f} & & & & \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Coefficients correspond to a 100-percentage-point "
        r"increase in the indicated weight, offset by Moral. The final column reports "
        r"the percentage-point change in giving for a 10-percentage-point increase. "
        r"Standard errors treat the estimated first-stage weights as fixed.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"A higher S weight predicts a lower share sent; the C-weight coefficient is "
        r"not statistically distinguishable from zero. We next estimate a pooled "
        r"multinomial logit for the classified post-choice reason, excluding N outcomes "
        r"and using Moral as the outcome baseline.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cell-specific HP weights and post-choice reasoning}",
        r"\label{tab:within_reasoning_prediction}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Weight & S vs M coefficient (SE) & $p$-value & C vs M coefficient (SE) & $p$-value \\",
        r"\midrule",
        *reasoning_rows,
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{N={int(reasoning_fit['n'])}; pseudo-$R^2="
        rf"{reasoning_fit['pseudo_r_squared']:.3f}$.}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Moral is the omitted outcome. The same controls as in "
        r"the allocation model are included. Parentheses contain participant-clustered "
        r"standard errors.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"Both HP weights predict non-Moral post-choice reasoning. This association is "
        r"stronger and more consistent than the allocation result.",
        "",
        r"\subsection{Average weight shifts and within-participant transitions}",
        "",
        r"For participants observed in both relevant cells, changes are measured "
        r"relative to KW Control. The predicted allocation change due to weights is "
        r"$\widehat\beta_S\Delta\bar w_S+\widehat\beta_C\Delta\bar w_C$, using the "
        r"pooled allocation slopes.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Average shifts in HP weights and allocation}",
        r"\label{tab:within_weight_shifts}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Comparison & $\Delta M$ & $\Delta S$ & $\Delta C$ & Predicted $\Delta y$ & Actual $\Delta y$ & N \\",
        r"\midrule",
        *shift_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Weight and allocation changes are in percentage points. "
        r"The decomposition is descriptive and is not a causal mediation estimate.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"The KW-to-LT weight shift predicts a 0.9 percentage-point reduction in giving, "
        r"while giving rises by 4.9 points. The KW-Control-to-KW-Market weight shift "
        r"predicts a 3.4-point reduction, compared with an observed 10.2-point reduction.",
        "",
        r"Common Moral-weight terciles use the usable KW-Control distribution "
        rf"(cutoffs {cutoffs['lower_cutoff']:.3f} and {cutoffs['upper_cutoff']:.3f}); "
        r"the same numerical cutoffs are applied to each destination cell.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Transitions across common Moral-weight terciles}",
        r"\label{tab:within_moral_transitions}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Origin tercile & Destination low & Destination middle & Destination high & Row N \\",
        r"\midrule",
        *transition_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: Entries are row percentages, with counts in parentheses. "
        r"Rows are KW-Control terciles. Only participants with usable M/S/C weights in "
        r"both cells are included.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"All within-subject results are descriptive, in-sample associations. The "
        r"weights are estimated from the HP classifications rather than directly "
        r"observed, and the treatment decomposition does not isolate a causal mechanism.",
    ]


def write_tex_document(
    regressions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    behavior_cv: pd.DataFrame,
) -> None:
    regression = regressions.set_index("term")
    bootstrap_by_term = bootstrap.set_index("term")
    pooled_cv = behavior_cv[behavior_cv["scope"] == "pooled"].set_index("model")
    labels = {
        "b_M_vs_N": r"$b_{Mi}$: Moral vs. No clear",
        "b_S_vs_N": r"$b_{Si}$: Self-interest vs. No clear",
        "b_C_vs_N": r"$b_{Ci}$: Cooperation vs. No clear",
    }
    table_rows = []
    for term in ["b_M_vs_N", "b_S_vs_N", "b_C_vs_N"]:
        row = regression.loc[term]
        boot = bootstrap_by_term.loc[term]
        effect_one_sd = row["coefficient"] * row["predictor_std_dev"]
        p_value = (
            r"$<0.001$"
            if row["p_value"] < 0.001
            else f"{row['p_value']:.3f}"
        )
        table_rows.append(
            f"{labels[term]} & {row['coefficient']:.3f} & "
            f"{row['std_error_clustered_pid']:.3f} & {p_value} & "
            f"[{boot['bootstrap_ci95_low_percentile']:.3f}, "
            f"{boot['bootstrap_ci95_high_percentile']:.3f}] & {effect_one_sd:+.3f} \\\\"
        )
    cell_only = pooled_cv.loc["cell_only"]
    full = pooled_cv.loc["cell_plus_participant_effects"]
    incremental_r2 = full["out_of_sample_r_squared"] - cell_only["out_of_sample_r_squared"]
    tex = "\n".join([
        r"\documentclass{article}",
        "",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\usepackage[hidelinks]{hyperref}",
        "",
        r"\title{Participant-Level HP Representation Effects}",
        r"\author{}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Between-subject analysis}",
        "",
        r"\subsection{Four-category method}",
        "",
        r"Each participant-frame (one participant under one set of instructions) "
        r"provides four HP texts, one for each hypothetical allocation "
        r"$a\in\{4,6,8,12\}$. Each text is classified as Moral (M), Self-interest "
        r"(S), Cooperation (C), or No clear justification (N). We estimate a multinomial "
        r"model with allocation-specific and participant-specific terms:",
        "",
        r"\begin{equation}",
        r"    \log\frac{\Pr(K_{ia}=k)}{\Pr(K_{ia}=N)}=\alpha_{ka}+b_{ki},",
        r"    \qquad k\in\{M,S,C\},",
        r"\end{equation}",
        "",
        r"where the participant-frame effects $b_{ki}$ are regularized toward zero. "
        r"Cross-validation selects the strength of this regularization. The allocation "
        r"effects $\alpha_{ka}$ absorb systematic differences across the four hypothetical "
        r"allocations; the second stage uses the estimated "
        r"$b_{Mi}$, $b_{Si}$, and $b_{Ci}$ directly. These effects measure participant "
        r"deviations in category log odds relative to No clear justification, net of "
        r"differences across the four hypothetical allocations.",
        "",
        r"We regress the participant's DG share sent on the three $b_{ki}$ effects, "
        r"controlling for indicators for LT Control, KW Market, and LT Market, with "
        r"KW Control omitted. Standard errors are "
        r"clustered by participant. Confidence intervals refit both stages in 500 "
        r"bootstrap samples drawn at the participant level, holding the regularization "
        r"strength at its cross-validated value.",
        "",
        r"\subsection{Four-category results}",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{HP participant effects and DG share sent}",
        r"\label{tab:hp_weights_dg}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Participant effect & Coefficient & Clustered SE & $p$-value & Bootstrap 95\% CI "
        r"& Effect of +1 SD \\",
        r"\midrule",
        *table_rows,
        r"\midrule",
        f"Observations & {int(regression.loc['const', 'n_frames'])} & & & & \\\\",
        f"$R^2$ & {regression.loc['const', 'r_squared']:.3f} & & & & \\\\",
        r"Four game-by-condition cells & Yes & & & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize Notes: No clear justification is the first-stage reference "
        r"category. Coefficients give the change in share sent for a one-unit increase "
        r"in the indicated participant log-odds effect. The final column rescales this "
        r"change to one sample standard deviation.",
        r"\end{flushleft}",
        r"\end{table}",
        "",
        r"The coefficients report predictive associations between the direct "
        r"participant-category effects and DG share sent; they are not causal effects.",
        "",
        f"In grouped 10-fold cross-validation, adding the participant effects changes "
        f"out-of-sample "
        f"$R^2$ from {cell_only['out_of_sample_r_squared']:.3f} to "
        rf"{full['out_of_sample_r_squared']:.3f} ($\Delta R^2={incremental_r2:.3f}$) "
        f"and changes RMSE from {cell_only['rmse']:.4f} to {full['rmse']:.4f}.",
        *moral_baseline_tex_section(),
        *heterogeneity_tex_section(),
        *within_subject_tex_section(),
        "",
        r"\end{document}",
        "",
    ])
    atomic_write_text(TEX_OUTPUT, tex)


def regenerate_tex() -> None:
    required = [
        OUT / "second_stage_regressions.csv",
        OUT / "bootstrap_summary.csv",
        OUT / "second_stage_grouped_cv.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot regenerate TeX; missing: "
            + ", ".join(str(path) for path in missing)
        )
    write_tex_document(
        pd.read_csv(required[0]),
        pd.read_csv(required[1]),
        pd.read_csv(required[2]),
    )
    print(f"Wrote {TEX_OUTPUT}")


def run_bootstrap(reps: int, checkpoint_every: int = 10) -> None:
    if reps < 1:
        raise ValueError("--reps must be positive.")
    progress = read_progress()
    if not progress.get("fit_complete"):
        raise RuntimeError("Run fit successfully before bootstrap.")
    data = load_and_validate()
    source_hash = file_sha256(INPUT)
    lambda_grid = [float(value) for value in progress["lambda_grid"]]
    fingerprint = model_fingerprint(source_hash, lambda_grid)
    if progress.get("model_fingerprint") != fingerprint:
        raise RuntimeError("Input or model specification changed after the fit stage.")
    selected_lambda = float(progress["selected_lambda"])
    draws_path = OUT / "bootstrap_draws.csv"
    if draws_path.exists():
        draws = pd.read_csv(draws_path)
        completed = set(draws["replicate"].astype(int).unique())
    else:
        draws = pd.DataFrame()
        completed = set()
    pending = [replicate for replicate in range(1, reps + 1) if replicate not in completed]
    write_progress(
        stage="bootstrap",
        bootstrap_target=reps,
        bootstrap_completed=len(completed),
        analysis_complete=False,
    )
    new_records: list[dict] = []
    for position, replicate in enumerate(pending, start=1):
        new_records.extend(bootstrap_one(data, selected_lambda, replicate))
        if position % checkpoint_every == 0 or position == len(pending):
            additions = pd.DataFrame(new_records)
            draws = pd.concat([draws, additions], ignore_index=True)
            draws = draws.sort_values(["replicate", "term"]).drop_duplicates(
                ["replicate", "term"], keep="last"
            )
            atomic_write_csv(draws, draws_path)
            new_records = []
            completed_count = draws["replicate"].nunique()
            write_progress(
                stage="bootstrap",
                bootstrap_target=reps,
                bootstrap_completed=int(completed_count),
                last_bootstrap_replicate=int(replicate),
            )
    regressions = pd.read_csv(OUT / "second_stage_regressions.csv")
    eligible = draws[draws["replicate"] <= reps]
    summary = summarize_bootstrap(eligible, regressions)
    atomic_write_csv(summary, OUT / "bootstrap_summary.csv")
    add_bootstrap_to_summary(summary)
    behavior_cv = pd.read_csv(OUT / "second_stage_grouped_cv.csv")
    write_tex_document(regressions, summary, behavior_cv)
    write_progress(
        stage="complete",
        bootstrap_target=reps,
        bootstrap_completed=int(eligible["replicate"].nunique()),
        analysis_complete=True,
    )


def show_status() -> None:
    if not PROGRESS.exists():
        print("No participant-representation analysis has been started.")
        return
    print(PROGRESS.read_text(encoding="utf-8"))
    for name in [
        "first_stage_lambda_cv.csv",
        "participant_frame_effects.csv",
        "second_stage_regressions.csv",
        "second_stage_grouped_cv.csv",
        "bootstrap_draws.csv",
        "bootstrap_summary.csv",
        "analysis_summary.txt",
    ]:
        path = OUT / name
        print(f"{name}: {'present' if path.exists() else 'missing'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["fit", "bootstrap", "all", "status", "tex"]
    )
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--lambda-grid", default=",".join(str(x) for x in LAMBDA_GRID_DEFAULT))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = [float(value.strip()) for value in args.lambda_grid.split(",") if value.strip()]
    if any(value <= 0 for value in lambda_grid) or not lambda_grid:
        raise ValueError("All lambda-grid values must be positive.")
    if args.command in ["fit", "all"]:
        run_fit(lambda_grid)
    if args.command in ["bootstrap", "all"]:
        run_bootstrap(args.reps, checkpoint_every=args.checkpoint_every)
    if args.command == "status":
        show_status()
    if args.command == "tex":
        regenerate_tex()


if __name__ == "__main__":
    main()
