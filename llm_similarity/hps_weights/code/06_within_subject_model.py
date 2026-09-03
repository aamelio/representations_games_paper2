"""Within-subject HP component model and period-specific predictions.

First stage (Moral omitted; No-clear HP rows excluded):

  log Pr(K=S or C) / Pr(K=M)
    = allocation FE + Market + KW + Market x KW + participant effect.

Participant effects receive the same Gaussian shrinkage used in the
between-subject analysis. The penalty is selected by holding out HP anchors
within participant-period. Second-stage allocation and reasoning models are
estimated separately for the first and second played game.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
WITHIN = ROOT / "within_subject"
INPUT = WITHIN / "data" / "within_hp_panel_classified.csv"
OUT = WITHIN / "output"
PROGRESS = OUT / "progress.json"

CATEGORIES = ["Moral", "Self-interest", "Mutual Benefit / Cooperation"]
MODELED = CATEGORIES[1:]
SHORT = {
    "Moral": "M",
    "Self-interest": "S",
    "Mutual Benefit / Cooperation": "C",
}
ANCHORS = [4.0, 6.0, 8.0, 12.0]
LAMBDA_GRID = [
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
]


@dataclass
class FirstStageFit:
    fixed: np.ndarray
    participant: np.ndarray
    success: bool
    iterations: int
    objective: float
    gradient_max: float
    message: str


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not INPUT.exists():
        raise FileNotFoundError(
            "The within-subject HP classification must be finalized first."
        )
    full = pd.read_csv(INPUT)
    full["category_num"] = pd.to_numeric(
        full["category_num"], errors="raise"
    ).astype(int)
    full["actual_reason_category_num"] = pd.to_numeric(
        full["actual_reason_category_num"], errors="coerce"
    ).astype("Int64")
    full["subject_id"] = full["subject_id"].astype(str)
    full["anchor_allocation_kept"] = pd.to_numeric(
        full["anchor_allocation_kept"], errors="raise"
    )
    substantive = full[full["category"].isin(CATEGORIES)].copy()
    usable_subjects = substantive["subject_id"].unique()
    substantive = substantive[
        substantive["subject_id"].isin(usable_subjects)
    ].reset_index(drop=True)
    summary = {
        "n_hp_rows_all": int(len(full)),
        "n_hp_rows_substantive": int(len(substantive)),
        "n_hp_rows_no_clear": int((full["category_num"] == 0).sum()),
        "n_subjects_all": int(full["subject_id"].nunique()),
        "n_subjects_usable": int(substantive["subject_id"].nunique()),
        "n_subjects_all_no_clear": int(
            full["subject_id"].nunique()
            - substantive["subject_id"].nunique()
        ),
        "n_elicitations_all": int(
            full[["subject_id", "trial_order"]].drop_duplicates().shape[0]
        ),
    }
    return full, substantive, summary


def fixed_design(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    anchor = data["anchor_allocation_kept"].astype(float)
    market = data["Market"].astype(float)
    kw = data["kw_dummy"].astype(float)
    names = [
        "Intercept",
        "Anchor=6",
        "Anchor=8",
        "Anchor=12",
        "Market",
        "KW",
        "Market_x_KW",
    ]
    design = np.column_stack([
        np.ones(len(data)),
        (anchor == 6).astype(float),
        (anchor == 8).astype(float),
        (anchor == 12).astype(float),
        market,
        kw,
        market * kw,
    ])
    return design, names


def prepare_first_stage(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    groups = sorted(data["subject_id"].unique())
    lookup = {group: index for index, group in enumerate(groups)}
    group_codes = data["subject_id"].map(lookup).to_numpy(dtype=int)
    design, fixed_names = fixed_design(data)
    category_lookup = {
        category: index for index, category in enumerate(CATEGORIES)
    }
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
    n_logits = len(MODELED)
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
        fixed, participant = unpack(parameters)
        nonbase = design @ fixed + participant[group_codes]
        logits = np.column_stack([np.zeros(len(outcome)), nonbase])
        denominator = logsumexp(logits, axis=1)
        objective = float(
            np.sum(denominator - logits[np.arange(len(outcome)), outcome])
        )
        objective += 0.5 * penalty * float(np.sum(participant ** 2))
        probability = np.exp(logits - denominator[:, None])
        residual = probability[:, 1:].copy()
        for code in range(1, len(CATEGORIES)):
            residual[:, code - 1] -= outcome == code
        fixed_gradient = design.T @ residual
        participant_gradient = np.zeros_like(participant)
        np.add.at(participant_gradient, group_codes, residual)
        participant_gradient += penalty * participant
        gradient = np.concatenate([
            fixed_gradient.ravel(),
            participant_gradient.ravel(),
        ])
        return objective, gradient

    result = minimize(
        objective_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-6},
    )
    fixed, participant = unpack(result.x)
    return FirstStageFit(
        fixed=fixed,
        participant=participant,
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
    participant = np.zeros((len(group_codes), len(MODELED)))
    observed = group_codes >= 0
    participant[observed] = fit.participant[group_codes[observed]]
    nonbase = design @ fit.fixed + participant
    logits = np.column_stack([np.zeros(len(design)), nonbase])
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def make_cv_folds(data: pd.DataFrame) -> np.ndarray:
    subject_order = {
        subject: index
        for index, subject in enumerate(sorted(data["subject_id"].unique()))
    }
    anchor_order = {anchor: index for index, anchor in enumerate(ANCHORS)}
    subject_index = data["subject_id"].map(subject_order).to_numpy(dtype=int)
    period_index = data["trial_order"].astype(int).to_numpy() - 1
    anchor_index = (
        data["anchor_allocation_kept"].map(anchor_order).to_numpy(dtype=int)
    )
    return (subject_index + period_index + anchor_index) % 4


def tune_penalty(data: pd.DataFrame) -> pd.DataFrame:
    folds = make_cv_folds(data)
    records = []
    for fold in range(4):
        train = data.loc[folds != fold].copy()
        test = data.loc[folds == fold].copy()
        train_x, train_groups, train_y, _, group_names = prepare_first_stage(
            train
        )
        group_lookup = {group: index for index, group in enumerate(group_names)}
        test_groups = (
            test["subject_id"].map(group_lookup).fillna(-1).to_numpy(dtype=int)
        )
        test_x, _ = fixed_design(test)
        category_lookup = {
            category: index for index, category in enumerate(CATEGORIES)
        }
        test_y = test["category"].map(category_lookup).to_numpy(dtype=int)
        initial = None
        for penalty in sorted(LAMBDA_GRID, reverse=True):
            fit = fit_penalized_multinomial(
                train_x,
                train_groups,
                train_y,
                penalty,
                initial=initial,
            )
            initial = np.concatenate([
                fit.fixed.ravel(),
                fit.participant.ravel(),
            ])
            probability = predict_probabilities(fit, test_x, test_groups)
            chosen = probability[np.arange(len(test_y)), test_y]
            records.append({
                "fold": fold + 1,
                "penalty_lambda": penalty,
                "n_train": len(train),
                "n_test": len(test),
                "n_unseen_test_subjects": int((test_groups < 0).sum()),
                "log_loss": float(
                    -np.mean(np.log(np.clip(chosen, 1e-15, 1.0)))
                ),
                "accuracy": float(
                    (probability.argmax(axis=1) == test_y).mean()
                ),
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


def fixed_effect_table(
    fit: FirstStageFit, fixed_names: list[str]
) -> pd.DataFrame:
    rows = []
    for fixed_index, term in enumerate(fixed_names):
        for category_index, category in enumerate(MODELED):
            rows.append({
                "term": term,
                "category_vs_moral": SHORT[category],
                "log_odds_coefficient": fit.fixed[
                    fixed_index, category_index
                ],
                "odds_ratio": np.exp(
                    fit.fixed[fixed_index, category_index]
                ),
            })
    return pd.DataFrame(rows)


def treatment_cell_probabilities(fit: FirstStageFit) -> pd.DataFrame:
    """Average fitted probabilities over anchors at zero participant effect."""
    cells = pd.DataFrame([
        {"treatment_cell": "lt_control", "Market": 0, "kw_dummy": 0},
        {"treatment_cell": "kw_control", "Market": 0, "kw_dummy": 1},
        {"treatment_cell": "lt_market", "Market": 1, "kw_dummy": 0},
        {"treatment_cell": "kw_market", "Market": 1, "kw_dummy": 1},
    ])
    grid = cells.assign(_key=1).merge(
        pd.DataFrame({"anchor_allocation_kept": ANCHORS, "_key": 1}),
        on="_key",
        how="inner",
    ).drop(columns="_key")
    grid["market_x_kw"] = grid["Market"] * grid["kw_dummy"]
    design, _ = fixed_design(grid)
    nonbase = design @ fit.fixed
    logits = np.column_stack([np.zeros(len(grid)), nonbase])
    probability = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    for category_index, category in enumerate(["M", "S", "C"]):
        grid[f"probability_{category}"] = probability[:, category_index]
    return grid.groupby("treatment_cell", as_index=False, observed=True)[
        ["probability_M", "probability_S", "probability_C"]
    ].mean()


def build_components(
    full: pd.DataFrame,
    fit: FirstStageFit,
    fixed_names: list[str],
    group_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = pd.DataFrame({
        "subject_id": group_names,
        "subject_effect_S": fit.participant[:, 0],
        "subject_effect_C": fit.participant[:, 1],
    })
    elicitation_columns = [
        "subject_id",
        "PROLIFIC_PID",
        "source_study",
        "game",
        "Market",
        "kw_dummy",
        "market_x_kw",
        "treatment_cell",
        "trial_order",
        "actual_allocation_kept",
        "actual_share_sent",
        "actual_reason",
        "actual_reason_category",
        "actual_reason_category_num",
    ]
    elicitation = full[elicitation_columns].drop_duplicates(
        ["subject_id", "trial_order"]
    )
    elicitation = elicitation.merge(
        participant, on="subject_id", how="inner", validate="many_to_one"
    )
    fixed_lookup = {
        name: index for index, name in enumerate(fixed_names)
    }
    for category_index, category in enumerate(["S", "C"]):
        elicitation[f"treatment_effect_{category}"] = (
            elicitation["Market"].astype(float)
            * fit.fixed[fixed_lookup["Market"], category_index]
            + elicitation["kw_dummy"].astype(float)
            * fit.fixed[fixed_lookup["KW"], category_index]
            + elicitation["market_x_kw"].astype(float)
            * fit.fixed[fixed_lookup["Market_x_KW"], category_index]
        )

    anchor_grid = elicitation.assign(_key=1).merge(
        pd.DataFrame({"anchor_allocation_kept": ANCHORS, "_key": 1}),
        on="_key",
        how="inner",
    ).drop(columns="_key")
    design, _ = fixed_design(anchor_grid)
    group_lookup = {group: index for index, group in enumerate(group_names)}
    group_codes = anchor_grid["subject_id"].map(group_lookup).to_numpy(
        dtype=int
    )
    probability = predict_probabilities(fit, design, group_codes)
    for category_index, category in enumerate(["M", "S", "C"]):
        anchor_grid[f"fitted_probability_{category}"] = probability[
            :, category_index
        ]
    weights = anchor_grid.groupby(
        elicitation_columns
        + [
            "subject_effect_S",
            "subject_effect_C",
            "treatment_effect_S",
            "treatment_effect_C",
        ],
        as_index=False,
        observed=True,
        dropna=False,
    )[
        [
            "fitted_probability_M",
            "fitted_probability_S",
            "fitted_probability_C",
        ]
    ].mean()
    return participant, weights


def predictor_design(
    data: pd.DataFrame, component: str
) -> pd.DataFrame:
    design = pd.DataFrame(
        {
            "const": 1.0,
            "market_control_study": (
                data["source_study"] == "market_control"
            ).astype(float),
        },
        index=data.index,
    )
    if component in {"subject", "full"}:
        design["subject_effect_S"] = data["subject_effect_S"].astype(float)
        design["subject_effect_C"] = data["subject_effect_C"].astype(float)
    if component in {"treatment", "full"}:
        design["treatment_effect_S"] = data[
            "treatment_effect_S"
        ].astype(float)
        design["treatment_effect_C"] = data[
            "treatment_effect_C"
        ].astype(float)
    return design


def allocation_models(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tests = []
    for period in [1, 2]:
        data = weights[weights["trial_order"] == period].copy()
        fitted = {}
        for component in ["base", "subject", "treatment", "full"]:
            design = predictor_design(data, component)
            model = sm.OLS(data["actual_share_sent"], design).fit(
                cov_type="HC3"
            )
            fitted[component] = model
            confidence = model.conf_int()
            for term in model.params.index:
                rows.append({
                    "period": period,
                    "model": component,
                    "term": term,
                    "coefficient": model.params[term],
                    "std_error_hc3": model.bse[term],
                    "p_value": model.pvalues[term],
                    "ci95_low": confidence.loc[term, 0],
                    "ci95_high": confidence.loc[term, 1],
                    "n": int(model.nobs),
                    "r_squared": model.rsquared,
                    "rmse": float(np.sqrt(np.mean(model.resid ** 2))),
                })
        full = fitted["full"]
        for component, terms in {
            "subject": ["subject_effect_S", "subject_effect_C"],
            "treatment": ["treatment_effect_S", "treatment_effect_C"],
        }.items():
            restriction = np.zeros((2, len(full.params)))
            for row_index, term in enumerate(terms):
                restriction[row_index, list(full.params.index).index(term)] = 1
            test = full.wald_test(restriction, scalar=True)
            reduced = fitted[
                "treatment" if component == "subject" else "subject"
            ]
            tests.append({
                "period": period,
                "component": component,
                "chi2_statistic": float(np.asarray(test.statistic).item()),
                "degrees_of_freedom": 2,
                "p_value": float(test.pvalue),
                "incremental_r_squared": full.rsquared - reduced.rsquared,
            })
    return pd.DataFrame(rows), pd.DataFrame(tests)


def fit_reasoning_model(
    data: pd.DataFrame, component: str
):
    design = predictor_design(data, component)
    outcome = data["actual_reason_category"].map(
        {category: index for index, category in enumerate(CATEGORIES)}
    )
    return sm.MNLogit(outcome, design).fit(
        method="newton", maxiter=200, disp=False
    )


def reasoning_models(
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    tests = []
    performance = []
    for period in [1, 2]:
        data = weights[
            (weights["trial_order"] == period)
            & weights["actual_reason_category"].isin(CATEGORIES)
        ].copy()
        fitted = {}
        for component in ["base", "subject", "treatment", "full"]:
            model = fit_reasoning_model(data, component)
            fitted[component] = model
            probability = np.asarray(model.predict())
            outcome = data["actual_reason_category"].map(
                {category: index for index, category in enumerate(CATEGORIES)}
            ).to_numpy(dtype=int)
            chosen = probability[np.arange(len(outcome)), outcome]
            performance.append({
                "period": period,
                "model": component,
                "n": len(data),
                "log_loss": float(
                    -np.mean(np.log(np.clip(chosen, 1e-15, 1.0)))
                ),
                "accuracy": float(
                    (probability.argmax(axis=1) == outcome).mean()
                ),
                "pseudo_r_squared": float(model.prsquared),
            })
            if component == "full":
                for outcome_column, outcome_category in enumerate(["S", "C"]):
                    for term in model.params.index:
                        rows.append({
                            "period": period,
                            "outcome_vs_moral": outcome_category,
                            "term": term,
                            "coefficient": model.params.loc[
                                term, outcome_column
                            ],
                            "std_error": model.bse.loc[term, outcome_column],
                            "p_value": model.pvalues.loc[
                                term, outcome_column
                            ],
                            "n": int(model.nobs),
                            "pseudo_r_squared": float(model.prsquared),
                        })
        full = fitted["full"]
        for component, reduced_name in [
            ("subject", "treatment"),
            ("treatment", "subject"),
        ]:
            reduced = fitted[reduced_name]
            statistic = 2.0 * (full.llf - reduced.llf)
            degrees = int(full.df_model - reduced.df_model)
            tests.append({
                "period": period,
                "component": component,
                "likelihood_ratio_chi2": statistic,
                "degrees_of_freedom": degrees,
                "p_value": chi2.sf(statistic, degrees),
                "incremental_pseudo_r_squared": (
                    full.prsquared - reduced.prsquared
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(tests), pd.DataFrame(performance)


def run_fit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, substantive, summary = load_data()
    lambda_cv = tune_penalty(substantive)
    selected_lambda = float(
        lambda_cv.loc[lambda_cv["selected"], "penalty_lambda"].iloc[0]
    )
    design, groups, outcome, fixed_names, group_names = prepare_first_stage(
        substantive
    )
    fit = fit_penalized_multinomial(
        design, groups, outcome, selected_lambda
    )
    participant, weights = build_components(
        full, fit, fixed_names, group_names
    )
    allocation, allocation_tests = allocation_models(weights)
    reasoning, reasoning_tests, reasoning_performance = reasoning_models(
        weights
    )

    atomic_write_csv(lambda_cv, OUT / "first_stage_lambda_cv.csv")
    atomic_write_csv(
        fixed_effect_table(fit, fixed_names),
        OUT / "first_stage_fixed_effects.csv",
    )
    atomic_write_csv(
        treatment_cell_probabilities(fit),
        OUT / "first_stage_treatment_cell_probabilities.csv",
    )
    atomic_write_csv(participant, OUT / "participant_effects.csv")
    atomic_write_csv(weights, OUT / "elicitation_components.csv")
    atomic_write_csv(allocation, OUT / "allocation_models.csv")
    atomic_write_csv(
        allocation_tests, OUT / "allocation_component_tests.csv"
    )
    atomic_write_csv(reasoning, OUT / "reasoning_models.csv")
    atomic_write_csv(reasoning_tests, OUT / "reasoning_component_tests.csv")
    atomic_write_csv(
        reasoning_performance, OUT / "reasoning_model_performance.csv"
    )
    summary.update({
        "selected_lambda": selected_lambda,
        "first_stage_converged": fit.success,
        "first_stage_iterations": fit.iterations,
        "first_stage_gradient_max": fit.gradient_max,
        "n_elicitations_usable": len(weights),
    })
    atomic_write_text(
        OUT / "sample_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        PROGRESS,
        json.dumps(
            {
                "stage": "fit_complete",
                "fit_complete": True,
                "analysis_complete": True,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def show_status() -> None:
    if not PROGRESS.exists():
        print("Within-subject model has not been fitted.")
        return
    print(PROGRESS.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
