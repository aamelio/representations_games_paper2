#!/usr/bin/env python3
"""Re-estimate HP representation weights on M/S/C with Moral as the baseline.

All "No clear justification" HP rows are removed. Participant-frames with no
remaining substantive classification are excluded because they contain no
participant-specific information with which to estimate an M/S/C prevalence.

Outputs are written to participant_representation_moral_baseline/. The
participant-cluster bootstrap is resumable and the completed analysis
regenerates ../../hps_weights.tex through the shared report generator in 04.

Usage:
    python 05_moral_baseline_excluding_no_clear.py fit
    python 05_moral_baseline_excluding_no_clear.py bootstrap --reps 500
    python 05_moral_baseline_excluding_no_clear.py all --reps 500
    python 05_moral_baseline_excluding_no_clear.py status
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
OUT = HERE / "participant_representation_moral_baseline"
PROGRESS = OUT / "progress.json"
BASE_SCRIPT = HERE / "04_participant_representation_model.py"
BASE_CATEGORY = "Moral"
MODELED_CATEGORIES = ["Self-interest", "Mutual Benefit / Cooperation"]
CATEGORIES = [BASE_CATEGORY, *MODELED_CATEGORIES]
CATEGORY_SHORT = {
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


def load_base_module():
    specification = importlib.util.spec_from_file_location("hp_weights_base", BASE_SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BASE = load_base_module()


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


def read_progress() -> dict:
    if not PROGRESS.exists():
        return {}
    return json.loads(PROGRESS.read_text(encoding="utf-8"))


def write_progress(**updates: object) -> dict:
    current = read_progress()
    current.update(updates)
    atomic_write_text(PROGRESS, json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def model_fingerprint(source_hash: str, lambda_grid: list[float]) -> str:
    specification = {
        "source_sha256": source_hash,
        "excluded_category": "No clear justification",
        "base_category": BASE_CATEGORY,
        "modeled_categories": MODELED_CATEGORIES,
        "hp_levels": HP_LEVELS,
        "lambda_grid": lambda_grid,
        "participant_unit": "PROLIFIC_PID x treatment x Market",
        "second_stage": "share_sent on S/C weights plus game-by-condition FE; M omitted",
    }
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_substantive_data() -> tuple[pd.DataFrame, dict]:
    full = BASE.load_and_validate()
    counts = full.groupby("frame_id", observed=True)["category"].apply(
        lambda values: int((values != "No clear justification").sum())
    )
    usable_frames = counts[counts > 0].index
    data = full[
        full["frame_id"].isin(usable_frames)
        & full["category"].isin(CATEGORIES)
    ].copy().reset_index(drop=True)
    frame_counts = data.groupby("frame_id", observed=True).size()
    summary = {
        "n_original_hp_rows": int(len(full)),
        "n_substantive_hp_rows": int(len(data)),
        "n_original_frames": int(full["frame_id"].nunique()),
        "n_usable_frames": int(data["frame_id"].nunique()),
        "n_excluded_all_no_clear_frames": int((counts == 0).sum()),
        "n_unique_pid": int(data["PROLIFIC_PID"].nunique()),
        "substantive_rows_per_frame": {
            str(int(key)): int(value)
            for key, value in frame_counts.value_counts().sort_index().items()
        },
    }
    if summary["n_substantive_hp_rows"] != 3059:
        raise ValueError(
            f"Expected 3,059 substantive HP rows; found {summary['n_substantive_hp_rows']}."
        )
    if summary["n_usable_frames"] != 1060:
        raise ValueError(
            f"Expected 1,060 usable frames; found {summary['n_usable_frames']}."
        )
    if summary["n_excluded_all_no_clear_frames"] != 140:
        raise ValueError(
            "Expected 140 frames containing only No-clear classifications; "
            f"found {summary['n_excluded_all_no_clear_frames']}."
        )
    return data, summary


def fixed_design(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    hp = data["hp"].astype(float)
    names = ["Intercept", "HP=6", "HP=8", "HP=12"]
    design = np.column_stack([
        np.ones(len(data)),
        (hp == 6).astype(float),
        (hp == 8).astype(float),
        (hp == 12).astype(float),
    ])
    return design, names


def prepare_first_stage(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    groups = sorted(data["frame_id"].astype(str).unique())
    lookup = {value: index for index, value in enumerate(groups)}
    group_codes = data["frame_id"].astype(str).map(lookup).to_numpy(dtype=int)
    design, fixed_names = fixed_design(data)
    category_lookup = {category: code for code, category in enumerate(CATEGORIES)}
    outcome = data["category"].map(category_lookup).to_numpy(dtype=int)
    return design, group_codes, outcome, fixed_names, groups


def fit_penalized_multinomial(
    design: np.ndarray,
    group_codes: np.ndarray,
    outcome: np.ndarray,
    penalty: float,
    initial: np.ndarray | None = None,
    maxiter: int = 500,
) -> FirstStageFit:
    n_fixed = design.shape[1]
    n_groups = int(group_codes.max()) + 1
    n_logits = len(MODELED_CATEGORIES)
    parameter_count = n_fixed * n_logits + n_groups * n_logits
    if initial is None or len(initial) != parameter_count:
        initial = np.zeros(parameter_count)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split = n_fixed * n_logits
        return (
            parameters[:split].reshape(n_fixed, n_logits),
            parameters[split:].reshape(n_groups, n_logits),
        )

    def objective_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        fixed_parameters, participant_parameters = unpack(parameters)
        nonbase = design @ fixed_parameters + participant_parameters[group_codes]
        logits = np.column_stack([np.zeros(len(outcome)), nonbase])
        log_denominator = logsumexp(logits, axis=1)
        objective = float(
            np.sum(log_denominator - logits[np.arange(len(outcome)), outcome])
        )
        objective += 0.5 * penalty * float(np.sum(participant_parameters ** 2))
        probabilities = np.exp(logits - log_denominator[:, None])
        residual = probabilities[:, 1:]
        for category_code in range(1, len(CATEGORIES)):
            residual[:, category_code - 1] -= outcome == category_code
        fixed_gradient = design.T @ residual
        participant_gradient = np.zeros_like(participant_parameters)
        np.add.at(participant_gradient, group_codes, residual)
        participant_gradient += penalty * participant_parameters
        gradient = np.concatenate([fixed_gradient.ravel(), participant_gradient.ravel()])
        return objective, gradient

    result = minimize(
        objective_gradient,
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
    design: np.ndarray,
    group_codes: np.ndarray,
) -> np.ndarray:
    participant = np.zeros((len(group_codes), len(MODELED_CATEGORIES)))
    observed = group_codes >= 0
    participant[observed] = fit.participant[group_codes[observed]]
    nonbase = design @ fit.fixed + participant
    logits = np.column_stack([np.zeros(len(design)), nonbase])
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def make_cv_folds(data: pd.DataFrame) -> np.ndarray:
    folds = np.full(len(data), -1, dtype=int)
    frame_order = {
        frame: index for index, frame in enumerate(sorted(data["frame_id"].unique()))
    }
    for frame, group in data.groupby("frame_id", observed=True):
        ordered_indices = group.sort_values("hp").index.to_numpy()
        for local_index, row_index in enumerate(ordered_indices):
            folds[row_index] = (frame_order[frame] + local_index) % 4
    if (folds < 0).any():
        raise AssertionError("Every substantive HP row must receive a CV fold.")
    return folds


def tune_penalty(data: pd.DataFrame, lambda_grid: list[float]) -> pd.DataFrame:
    folds = make_cv_folds(data)
    records = []
    for fold in range(4):
        train = data.loc[folds != fold].copy()
        test = data.loc[folds == fold].copy()
        train_x, train_groups, train_y, _, group_names = prepare_first_stage(train)
        group_lookup = {name: index for index, name in enumerate(group_names)}
        test_groups = (
            test["frame_id"].map(group_lookup).fillna(-1).to_numpy(dtype=int)
        )
        test_x, _ = fixed_design(test)
        category_lookup = {category: code for code, category in enumerate(CATEGORIES)}
        test_y = test["category"].map(category_lookup).to_numpy(dtype=int)
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
                "n_unseen_test_frames": int((test_groups < 0).sum()),
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
    rows = []
    for fixed_index, term in enumerate(fixed_names):
        for category_index, category in enumerate(MODELED_CATEGORIES):
            rows.append({
                "term": term,
                "category_vs_moral": category,
                "log_odds_coefficient": fit.fixed[fixed_index, category_index],
                "odds_ratio": np.exp(fit.fixed[fixed_index, category_index]),
            })
    return pd.DataFrame(rows)


def build_weights(
    data: pd.DataFrame,
    fit: FirstStageFit,
    group_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = data.groupby("frame_id", as_index=False, observed=True).agg(
        PROLIFIC_PID=("PROLIFIC_PID", "first"),
        treatment=("treatment", "first"),
        Market=("Market", "first"),
        allocation_kept=("allocation", "first"),
        substantive_classifications=("category", "size"),
    )
    metadata["share_sent"] = (12.0 - metadata["allocation_kept"]) / 12.0
    metadata["cell"] = metadata["treatment"] + "_" + metadata["Market"].map(
        {0: "control", 1: "market"}
    )
    anchors = pd.DataFrame({"hp": HP_LEVELS})
    grid = metadata.assign(_key=1).merge(
        anchors.assign(_key=1), on="_key", how="inner"
    ).drop(columns="_key")
    design, _ = fixed_design(grid)
    group_lookup = {name: index for index, name in enumerate(group_names)}
    group_codes = grid["frame_id"].map(group_lookup).to_numpy(dtype=int)
    probabilities = predict_probabilities(fit, design, group_codes)
    anchor_output = grid[[
        "frame_id", "PROLIFIC_PID", "treatment", "Market", "allocation_kept",
        "share_sent", "cell", "substantive_classifications", "hp",
    ]].copy()
    probability_columns = []
    for category_index, category in enumerate(CATEGORIES):
        column = f"fitted_probability_{CATEGORY_SHORT[category]}"
        probability_columns.append(column)
        anchor_output[column] = probabilities[:, category_index]
    weights = anchor_output.groupby(
        [
            "frame_id", "PROLIFIC_PID", "treatment", "Market", "allocation_kept",
            "share_sent", "cell", "substantive_classifications",
        ],
        as_index=False,
        observed=True,
    )[probability_columns].mean()
    weights["fitted_probability_sum"] = weights[probability_columns].sum(axis=1)
    return anchor_output, weights


def regression_design(weights: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "const": 1.0,
        "weight_S": weights["fitted_probability_S"].astype(float),
        "weight_C": weights["fitted_probability_C"].astype(float),
        "cell_lt_control": (weights["cell"] == "lt_control").astype(float),
        "cell_kw_market": (weights["cell"] == "kw_market").astype(float),
        "cell_lt_market": (weights["cell"] == "lt_market").astype(float),
    }, index=weights.index)


def fit_second_stage(weights: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    design = regression_design(weights)
    result = sm.OLS(weights["share_sent"].astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": weights["PROLIFIC_PID"], "use_correction": True},
    )
    confidence = result.conf_int(alpha=0.05)
    rows = []
    for term in result.params.index:
        rows.append({
            "model": "moral_baseline_excluding_no_clear",
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
            "omitted_category_weight": "M (Moral)",
            "cell_reference": "kw_control",
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
    baseline_design = full_design[:, [0, 3, 4, 5]]
    predictions = {
        "cell_only": np.full(len(weights), np.nan),
        "cell_plus_representation_weights": np.full(len(weights), np.nan),
    }
    rows = []
    for held_fold in range(n_folds):
        train = fold != held_fold
        test = ~train
        for model_name, design in [
            ("cell_only", baseline_design),
            ("cell_plus_representation_weights", full_design),
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
    increment = {
        "scope": "increment",
        "fold": np.nan,
        "model": "representation_weights_minus_cell_only",
        "n_test_frames": len(weights),
        "rmse": pooled.loc["cell_plus_representation_weights", "rmse"]
        - pooled.loc["cell_only", "rmse"],
        "mae": pooled.loc["cell_plus_representation_weights", "mae"]
        - pooled.loc["cell_only", "mae"],
        "out_of_sample_r_squared": pooled.loc[
            "cell_plus_representation_weights", "out_of_sample_r_squared"
        ] - pooled.loc["cell_only", "out_of_sample_r_squared"],
    }
    return pd.concat([output, pd.DataFrame([increment])], ignore_index=True)


def run_fit(lambda_grid: list[float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, sample_summary = load_substantive_data()
    source_hash = BASE.file_sha256(BASE.INPUT)
    fingerprint = model_fingerprint(source_hash, lambda_grid)
    existing = read_progress()
    bootstrap_path = OUT / "bootstrap_draws.csv"
    if bootstrap_path.exists() and existing.get("model_fingerprint") not in [None, fingerprint]:
        raise RuntimeError(
            "Existing bootstrap draws belong to a different model. Archive the output folder first."
        )
    atomic_write_text(
        OUT / "sample_summary.json",
        json.dumps(sample_summary, indent=2, sort_keys=True) + "\n",
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
    design, groups, outcome, fixed_names, group_names = prepare_first_stage(data)
    fit = fit_penalized_multinomial(design, groups, outcome, selected_lambda)
    if not fit.success and fit.gradient_max > 1e-4:
        raise RuntimeError(
            f"First-stage optimization failed: {fit.message}; "
            f"max |gradient|={fit.gradient_max:.3g}."
        )
    atomic_write_csv(
        fixed_effect_table(fit, fixed_names),
        OUT / "first_stage_fixed_effects.csv",
    )
    anchor_probabilities, weights = build_weights(data, fit, group_names)
    if not np.allclose(weights["fitted_probability_sum"], 1.0, atol=1e-10):
        raise AssertionError("M/S/C fitted probabilities do not sum to one.")
    atomic_write_csv(
        anchor_probabilities,
        OUT / "moral_baseline_anchor_probabilities.csv",
    )
    atomic_write_csv(
        weights,
        OUT / "participant_frame_category_weights.csv",
    )
    regressions, _ = fit_second_stage(weights)
    behavior_cv = grouped_behavior_cv(weights)
    atomic_write_csv(regressions, OUT / "second_stage_regressions.csv")
    atomic_write_csv(behavior_cv, OUT / "second_stage_grouped_cv.csv")
    write_progress(
        stage="fit_complete",
        fit_complete=True,
        selected_lambda=selected_lambda,
        final_converged=fit.success,
        final_iterations=fit.iterations,
        final_gradient_max=fit.gradient_max,
        n_substantive_hp_rows=len(data),
        n_frames=len(weights),
        n_unique_pid=weights["PROLIFIC_PID"].nunique(),
    )


def bootstrap_one(data: pd.DataFrame, penalty: float, replicate: int) -> list[dict]:
    sample = BASE.bootstrap_sample(data, replicate)
    design, groups, outcome, _, group_names = prepare_first_stage(sample)
    fit = fit_penalized_multinomial(
        design, groups, outcome, penalty, maxiter=350
    )
    _, weights = build_weights(sample, fit, group_names)
    _, result = fit_second_stage(weights)
    return [
        {
            "replicate": replicate,
            "term": term,
            "coefficient": float(result.params[term]),
            "first_stage_converged": fit.success,
            "first_stage_iterations": fit.iterations,
            "first_stage_gradient_max": fit.gradient_max,
            "n_frames": len(weights),
            "n_bootstrap_clusters": weights["PROLIFIC_PID"].nunique(),
        }
        for term in ["weight_S", "weight_C"]
    ]


def summarize_bootstrap(draws: pd.DataFrame, regressions: pd.DataFrame) -> pd.DataFrame:
    point = regressions.set_index("term")["coefficient"]
    rows = []
    for term, group in draws.groupby("term", observed=True):
        rows.append({
            "term": term,
            "point_estimate": point[term],
            "bootstrap_standard_error": group["coefficient"].std(ddof=1),
            "bootstrap_ci95_low_percentile": group["coefficient"].quantile(0.025),
            "bootstrap_ci95_high_percentile": group["coefficient"].quantile(0.975),
            "bootstrap_replicates": group["replicate"].nunique(),
            "failed_first_stage_fits": int((~group["first_stage_converged"]).sum()),
        })
    return pd.DataFrame(rows)


def regenerate_tex() -> None:
    original_out = BASE.OUT
    BASE.write_tex_document(
        pd.read_csv(original_out / "second_stage_regressions.csv"),
        pd.read_csv(original_out / "bootstrap_summary.csv"),
        pd.read_csv(original_out / "second_stage_grouped_cv.csv"),
    )


def run_bootstrap(reps: int, checkpoint_every: int) -> None:
    progress = read_progress()
    if not progress.get("fit_complete"):
        raise RuntimeError("Run fit successfully before bootstrap.")
    data, _ = load_substantive_data()
    source_hash = BASE.file_sha256(BASE.INPUT)
    lambda_grid = [float(value) for value in progress["lambda_grid"]]
    fingerprint = model_fingerprint(source_hash, lambda_grid)
    if progress.get("model_fingerprint") != fingerprint:
        raise RuntimeError("Input or model specification changed after the fit stage.")
    penalty = float(progress["selected_lambda"])
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
    new_rows = []
    for position, replicate in enumerate(pending, start=1):
        new_rows.extend(bootstrap_one(data, penalty, replicate))
        if position % checkpoint_every == 0 or position == len(pending):
            draws = pd.concat([draws, pd.DataFrame(new_rows)], ignore_index=True)
            draws = draws.sort_values(["replicate", "term"]).drop_duplicates(
                ["replicate", "term"], keep="last"
            )
            atomic_write_csv(draws, draws_path)
            new_rows = []
            write_progress(
                stage="bootstrap",
                bootstrap_target=reps,
                bootstrap_completed=int(draws["replicate"].nunique()),
                last_bootstrap_replicate=int(replicate),
            )
    eligible = draws[draws["replicate"] <= reps]
    regressions = pd.read_csv(OUT / "second_stage_regressions.csv")
    summary = summarize_bootstrap(eligible, regressions)
    atomic_write_csv(summary, OUT / "bootstrap_summary.csv")
    regenerate_tex()
    write_progress(
        stage="complete",
        bootstrap_target=reps,
        bootstrap_completed=int(eligible["replicate"].nunique()),
        analysis_complete=True,
    )


def show_status() -> None:
    if not PROGRESS.exists():
        print("No Moral-baseline analysis has been started.")
        return
    print(PROGRESS.read_text(encoding="utf-8"))
    for name in [
        "sample_summary.json",
        "first_stage_lambda_cv.csv",
        "participant_frame_category_weights.csv",
        "second_stage_regressions.csv",
        "second_stage_grouped_cv.csv",
        "bootstrap_draws.csv",
        "bootstrap_summary.csv",
    ]:
        print(f"{name}: {'present' if (OUT / name).exists() else 'missing'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "bootstrap", "all", "status"])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument(
        "--lambda-grid",
        default=",".join(str(value) for value in LAMBDA_GRID_DEFAULT),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = [
        float(value.strip())
        for value in args.lambda_grid.split(",")
        if value.strip()
    ]
    if not lambda_grid or any(value <= 0 for value in lambda_grid):
        raise ValueError("All lambda-grid values must be positive.")
    if args.command in ["fit", "all"]:
        run_fit(lambda_grid)
    if args.command in ["bootstrap", "all"]:
        run_bootstrap(args.reps, args.checkpoint_every)
    if args.command == "status":
        show_status()


if __name__ == "__main__":
    main()
