"""Test heterogeneity in HP-weight slopes across DG games and conditions.

This uses the Moral-baseline specification (M omitted; No-clear classifications
excluded). It estimates two models with game-by-condition fixed effects:

1. HP weights interacted with LT (reference: KW);
2. HP weights interacted with Market (reference: Control).

The participant-cluster bootstrap refits both the first-stage representation
model and the interacted second-stage models. Outputs are checkpointed so the
bootstrap can be resumed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OUT = ROOT / "output" / "heterogeneity"
PROGRESS = OUT / "progress.json"
WEIGHTS_PATH = (
    ROOT / "output" / "moral_baseline" / "participant_frame_category_weights.csv"
)
MORAL_PROGRESS = ROOT / "output" / "moral_baseline" / "progress.json"
BOOTSTRAP_SEED = 20260824


def load_moral_module():
    path = CODE_DIR / "02_moral_baseline_excluding_no_clear.py"
    spec = importlib.util.spec_from_file_location("moral_baseline_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MORAL = load_moral_module()


DIMENSIONS = {
    "game": {
        "base_group": "KW",
        "comparison_group": "LT",
    },
    "condition": {
        "base_group": "Control",
        "comparison_group": "Market",
    },
}


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_progress() -> dict:
    if not PROGRESS.exists():
        return {}
    return json.loads(PROGRESS.read_text(encoding="utf-8"))


def write_progress(**updates: object) -> dict:
    progress = read_progress()
    progress.update(updates)
    atomic_write_text(PROGRESS, json.dumps(progress, indent=2, sort_keys=True) + "\n")
    return progress


def model_fingerprint() -> str:
    source_hash = MORAL.BASE.file_sha256(MORAL.BASE.INPUT)
    specification = {
        "source_sha256": source_hash,
        "first_stage": "moral_baseline_excluding_no_clear",
        "second_stage": (
            "share_sent on S/C weights, game-condition FE, interactions"
        ),
        "dimensions": DIMENSIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()


def load_weights() -> pd.DataFrame:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Run 02_moral_baseline_excluding_no_clear.py fit first."
        )
    weights = pd.read_csv(WEIGHTS_PATH)
    expected_cells = {"kw_control", "lt_control", "kw_market", "lt_market"}
    if set(weights["cell"].unique()) != expected_cells:
        raise ValueError("Unexpected game-by-condition cells in the weights file.")
    columns = [
        "fitted_probability_M",
        "fitted_probability_S",
        "fitted_probability_C",
    ]
    if not np.allclose(weights[columns].sum(axis=1), 1.0):
        raise ValueError("M/S/C fitted probabilities do not sum to one.")
    return weights


def regression_design(weights: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dimension}")
    if dimension == "game":
        indicator = (weights["treatment"].str.lower() == "lt").astype(float)
        suffix = "LT"
    else:
        indicator = weights["Market"].astype(float)
        suffix = "Market"
    weight_s = weights["fitted_probability_S"].astype(float)
    weight_c = weights["fitted_probability_C"].astype(float)
    return pd.DataFrame(
        {
            "const": 1.0,
            "weight_S": weight_s,
            "weight_C": weight_c,
            "cell_lt_control": (weights["cell"] == "lt_control").astype(float),
            "cell_kw_market": (weights["cell"] == "kw_market").astype(float),
            "cell_lt_market": (weights["cell"] == "lt_market").astype(float),
            f"weight_S_x_{suffix}": weight_s * indicator,
            f"weight_C_x_{suffix}": weight_c * indicator,
        },
        index=weights.index,
    )


def linear_combination(result, terms: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    vector = np.array([terms.get(name, 0.0) for name in names])
    estimate = float(vector @ result.params.to_numpy())
    variance = float(vector @ result.cov_params().to_numpy() @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    z_value = estimate / standard_error if standard_error > 0 else np.nan
    p_value = (
        float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    )
    return {
        "estimate": estimate,
        "std_error": standard_error,
        "p_value": p_value,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
    }


def fit_dimension(
    weights: pd.DataFrame, dimension: str
) -> tuple[pd.DataFrame, dict, object]:
    design = regression_design(weights, dimension)
    result = sm.OLS(weights["share_sent"].astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": weights["PROLIFIC_PID"], "use_correction": True},
    )
    specification = DIMENSIONS[dimension]
    suffix = "LT" if dimension == "game" else "Market"
    rows = []
    for category in ["S", "C"]:
        weight_term = f"weight_{category}"
        interaction_term = f"weight_{category}_x_{suffix}"
        base = linear_combination(result, {weight_term: 1.0})
        comparison = linear_combination(
            result, {weight_term: 1.0, interaction_term: 1.0}
        )
        difference = linear_combination(result, {interaction_term: 1.0})
        row = {
            "dimension": dimension,
            "category_vs_moral": category,
            "base_group": specification["base_group"],
            "comparison_group": specification["comparison_group"],
            "n_frames": int(result.nobs),
            "n_unique_pid": int(weights["PROLIFIC_PID"].nunique()),
            "r_squared": float(result.rsquared),
        }
        for label, values in [
            ("base_slope", base),
            ("comparison_slope", comparison),
            ("difference", difference),
        ]:
            for statistic, value in values.items():
                row[f"{label}_{statistic}"] = value
        rows.append(row)

    parameter_names = list(result.params.index)
    restrictions = np.zeros((2, len(parameter_names)))
    restrictions[0, parameter_names.index(f"weight_S_x_{suffix}")] = 1.0
    restrictions[1, parameter_names.index(f"weight_C_x_{suffix}")] = 1.0
    joint = result.wald_test(restrictions, scalar=True)
    joint_test = {
        "dimension": dimension,
        "null_hypothesis": "S and C slope differences jointly equal zero",
        "chi2_statistic": float(np.asarray(joint.statistic).item()),
        "degrees_of_freedom": 2,
        "p_value": float(joint.pvalue),
    }
    return pd.DataFrame(rows), joint_test, result


def fit_all_dimensions(
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates = []
    joint_tests = []
    for dimension in DIMENSIONS:
        rows, joint, _ = fit_dimension(weights, dimension)
        estimates.append(rows)
        joint_tests.append(joint)
    return pd.concat(estimates, ignore_index=True), pd.DataFrame(joint_tests)


def run_fit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not MORAL_PROGRESS.exists():
        raise FileNotFoundError("The Moral-baseline fit progress file is missing.")
    moral_progress = json.loads(MORAL_PROGRESS.read_text(encoding="utf-8"))
    if not moral_progress.get("fit_complete"):
        raise RuntimeError("The Moral-baseline first-stage fit is incomplete.")
    weights = load_weights()
    estimates, joint_tests = fit_all_dimensions(weights)
    atomic_write_csv(estimates, OUT / "point_estimates.csv")
    atomic_write_csv(joint_tests, OUT / "joint_tests.csv")
    write_progress(
        stage="fit_complete",
        fit_complete=True,
        analysis_complete=False,
        model_fingerprint=model_fingerprint(),
        selected_lambda=float(moral_progress["selected_lambda"]),
        n_frames=len(weights),
        n_unique_pid=weights["PROLIFIC_PID"].nunique(),
    )


def bootstrap_one(
    data: pd.DataFrame, penalty: float, replicate: int
) -> list[dict]:
    sample = MORAL.BASE.bootstrap_sample(data, replicate)
    design, groups, outcome, _, group_names = MORAL.prepare_first_stage(sample)
    first_stage = MORAL.fit_penalized_multinomial(
        design, groups, outcome, penalty, maxiter=350
    )
    _, weights = MORAL.build_weights(sample, first_stage, group_names)
    point, _ = fit_all_dimensions(weights)
    rows = []
    for result in point.itertuples(index=False):
        for estimand in ["base_slope", "comparison_slope", "difference"]:
            rows.append(
                {
                    "replicate": replicate,
                    "dimension": result.dimension,
                    "category_vs_moral": result.category_vs_moral,
                    "estimand": estimand,
                    "estimate": getattr(result, f"{estimand}_estimate"),
                    "first_stage_converged": first_stage.success,
                    "first_stage_iterations": first_stage.iterations,
                    "first_stage_gradient_max": first_stage.gradient_max,
                    "n_frames": len(weights),
                    "n_bootstrap_clusters": weights[
                        "PROLIFIC_PID"
                    ].nunique(),
                }
            )
    return rows


def summarize_bootstrap(
    draws: pd.DataFrame, point: pd.DataFrame
) -> pd.DataFrame:
    point_long = point.melt(
        id_vars=["dimension", "category_vs_moral"],
        value_vars=[
            "base_slope_estimate",
            "comparison_slope_estimate",
            "difference_estimate",
        ],
        var_name="estimand",
        value_name="point_estimate",
    )
    point_long["estimand"] = point_long["estimand"].str.removesuffix(
        "_estimate"
    )
    rows = []
    keys = ["dimension", "category_vs_moral", "estimand"]
    for key, group in draws.groupby(keys, observed=True):
        rows.append(
            {
                "dimension": key[0],
                "category_vs_moral": key[1],
                "estimand": key[2],
                "bootstrap_standard_error": group["estimate"].std(ddof=1),
                "bootstrap_ci95_low_percentile": group["estimate"].quantile(
                    0.025
                ),
                "bootstrap_ci95_high_percentile": group["estimate"].quantile(
                    0.975
                ),
                "bootstrap_replicates": group["replicate"].nunique(),
                "failed_first_stage_fits": int(
                    (~group["first_stage_converged"]).sum()
                ),
            }
        )
    return point_long.merge(
        pd.DataFrame(rows), on=keys, validate="one_to_one"
    )


def write_summary(
    point: pd.DataFrame, bootstrap: pd.DataFrame, joint: pd.DataFrame
) -> None:
    boot = bootstrap.set_index(
        ["dimension", "category_vs_moral", "estimand"]
    )
    lines = [
        "Between-subject heterogeneity in HP-weight slopes",
        (
            "Moral baseline; No-clear classifications excluded; "
            "game-by-condition fixed effects."
        ),
        "",
    ]
    for dimension, specification in DIMENSIONS.items():
        lines.append(
            f"{dimension.title()}: {specification['comparison_group']} minus "
            f"{specification['base_group']}"
        )
        subset = point[point["dimension"] == dimension]
        for row in subset.itertuples(index=False):
            ci = boot.loc[(dimension, row.category_vs_moral, "difference")]
            lines.append(
                f"  {row.category_vs_moral} vs M: "
                f"{row.base_group} {row.base_slope_estimate:+.4f}; "
                f"{row.comparison_group} {row.comparison_slope_estimate:+.4f}; "
                f"difference {row.difference_estimate:+.4f} "
                f"(clustered p={row.difference_p_value:.4f}; "
                f"bootstrap 95% CI "
                f"[{ci.bootstrap_ci95_low_percentile:+.4f}, "
                f"{ci.bootstrap_ci95_high_percentile:+.4f}])."
            )
        joint_row = joint[joint["dimension"] == dimension].iloc[0]
        lines.append(
            f"  Joint interaction test: p={joint_row['p_value']:.4f}."
        )
        lines.append("")
    atomic_write_text(OUT / "analysis_summary.txt", "\n".join(lines))


def regenerate_tex() -> None:
    original_out = MORAL.BASE.OUT
    MORAL.BASE.write_tex_document(
        pd.read_csv(original_out / "second_stage_regressions.csv"),
        pd.read_csv(original_out / "bootstrap_summary.csv"),
        pd.read_csv(original_out / "second_stage_grouped_cv.csv"),
    )


def run_bootstrap(reps: int, checkpoint_every: int, jobs: int) -> None:
    if reps < 1:
        raise ValueError("--reps must be positive.")
    progress = read_progress()
    if not progress.get("fit_complete"):
        raise RuntimeError("Run the heterogeneity fit before the bootstrap.")
    if progress.get("model_fingerprint") != model_fingerprint():
        raise RuntimeError(
            "Input or heterogeneity specification changed after the fit."
        )
    data, _ = MORAL.load_substantive_data()
    penalty = float(progress["selected_lambda"])
    draws_path = OUT / "bootstrap_draws.csv"
    if draws_path.exists():
        draws = pd.read_csv(draws_path)
        completed = set(draws["replicate"].astype(int).unique())
    else:
        draws = pd.DataFrame()
        completed = set()
    pending = [
        replicate
        for replicate in range(1, reps + 1)
        if replicate not in completed
    ]
    write_progress(
        stage="bootstrap",
        bootstrap_target=reps,
        bootstrap_completed=len(completed),
        analysis_complete=False,
    )
    new_rows = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(bootstrap_one, data, penalty, replicate): replicate
            for replicate in pending
        }
        completed_futures = as_completed(futures)
        for position, future in enumerate(completed_futures, start=1):
            replicate = futures[future]
            new_rows.extend(future.result())
            if position % checkpoint_every == 0 or position == len(pending):
                draws = pd.concat(
                    [draws, pd.DataFrame(new_rows)], ignore_index=True
                )
                draws = draws.sort_values(
                    [
                        "replicate",
                        "dimension",
                        "category_vs_moral",
                        "estimand",
                    ]
                ).drop_duplicates(
                    [
                        "replicate",
                        "dimension",
                        "category_vs_moral",
                        "estimand",
                    ],
                    keep="last",
                )
                atomic_write_csv(draws, draws_path)
                new_rows = []
                write_progress(
                    stage="bootstrap",
                    bootstrap_target=reps,
                    bootstrap_completed=int(draws["replicate"].nunique()),
                    last_bootstrap_replicate=int(replicate),
                )
                print(
                    f"Completed {int(draws['replicate'].nunique())}/{reps} "
                    "bootstrap replicates."
                )
    eligible = draws[draws["replicate"] <= reps]
    point = pd.read_csv(OUT / "point_estimates.csv")
    joint = pd.read_csv(OUT / "joint_tests.csv")
    summary = summarize_bootstrap(eligible, point)
    atomic_write_csv(summary, OUT / "bootstrap_summary.csv")
    write_summary(point, summary, joint)
    regenerate_tex()
    write_progress(
        stage="complete",
        bootstrap_target=reps,
        bootstrap_completed=int(eligible["replicate"].nunique()),
        analysis_complete=True,
    )


def show_status() -> None:
    progress = read_progress()
    if not progress:
        print("No heterogeneity analysis has been started.")
        return
    print(json.dumps(progress, indent=2, sort_keys=True))
    for name in [
        "point_estimates.csv",
        "joint_tests.csv",
        "bootstrap_draws.csv",
        "bootstrap_summary.csv",
        "analysis_summary.txt",
    ]:
        print(
            f"{name}: {'present' if (OUT / name).exists() else 'missing'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "bootstrap", "status"])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "fit":
        run_fit()
    elif args.action == "bootstrap":
        run_bootstrap(args.reps, args.checkpoint_every, args.jobs)
    else:
        show_status()


if __name__ == "__main__":
    main()
