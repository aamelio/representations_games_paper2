"""07_player2_exhibit_split.py --- Split the Player 2 treatment-effects exhibit (E8).

Re-layout (no new statistics) of appendix_player2_treatment_effects.png, per the
coauthor decision of 2026-07-13 (AA + NG email thread): the main text shows only
Player 2 *hypothetical* behavior --- the clean partial-equilibrium outcomes ---
before the benchmark regressions; the category-share shifts move to the Player 2
appendix.

Outputs (output/figures/):
  - player2_hypothetical_treatment_effects.png  (main text: HP outcome effects
    at the one-third reference action + treatment-level means of HP actions)
  - appendix_player2_category_shifts.png        (appendix: category-share shifts)

Every panel is produced by the same compute/plot functions of
01_control_treatment_and_appendix_figures.py, so panel content is identical to
the corresponding rows of the original three-row figure.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "control_figures", ROOT / "01_control_treatment_and_appendix_figures.py"
)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


def build_main_text_hypothetical_figure(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    outcome_effects = cf.compute_outcome_model_effects(p1, p2)

    # Top row only (treatment effects on hypothetical outcomes); the
    # treatment-level means of Player 2 hypothetical actions (former bottom row)
    # were dropped per the coauthor request of 2026-07-19.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9))

    for col_idx, comparison in enumerate(cf.COMPARISONS):
        slug = comparison["slug"]
        title = comparison["title"]
        cf.plot_outcome_panel(
            axes[col_idx],
            outcome_effects[
                (outcome_effects["player"] == "player2")
                & (outcome_effects["comparison_slug"] == slug)
            ].copy(),
            cf.PLAYER_GAMES["player2"],
            f"{title}: Player 2 HP outcome",
        )

    axes[0].set_ylabel("Treatment effect on HP outcome (pp)")

    fig.tight_layout()
    cf.save_figure(fig, "player2_hypothetical_treatment_effects.png")


def build_appendix_category_shift_figure(p2: pd.DataFrame) -> None:
    p2_effects = cf.compute_representation_effects(
        p2, cf.P2_CATEGORIES, cf.PLAYER_GAMES["player2"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for col_idx, comparison in enumerate(cf.COMPARISONS):
        slug = comparison["slug"]
        title = comparison["title"]
        cf.plot_representation_panel(
            axes[col_idx],
            p2_effects[p2_effects["comparison_slug"] == slug].copy(),
            cf.P2_CATEGORIES,
            cf.P2_CATEGORY_LABELS,
            cf.PLAYER_GAMES["player2"],
            f"{title}: Player 2 representations",
        )

    axes[0].set_ylabel("Difference in category share (pp)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    cf.save_figure(fig, "appendix_player2_category_shifts.png")


def main() -> None:
    cf.set_plot_style()
    p1 = cf.load_player1()
    p2 = cf.load_player2()
    build_main_text_hypothetical_figure(p1, p2)
    build_appendix_category_shift_figure(p2)
    print(
        "Wrote output/figures/player2_hypothetical_treatment_effects.png "
        "and output/figures/appendix_player2_category_shifts.png"
    )


if __name__ == "__main__":
    main()
