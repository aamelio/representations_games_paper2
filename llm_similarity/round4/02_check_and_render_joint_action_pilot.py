#!/usr/bin/env python3
"""Validate and render the 60-vignette joint-action pilot.

Input:  01_joint_action_pilot.csv
Outputs: joint_action_pilot_readable.md, joint_action_pilot_checks.txt
"""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "01_joint_action_pilot.csv"
READABLE = HERE / "joint_action_pilot_readable.md"
CHECKS = HERE / "joint_action_pilot_checks.txt"

STRUCTURE_ORDER = ["DG", "UG", "TG"]
CATEGORY_ORDER = ["Moral", "Self-interest", "Cooperation"]
ACTION_ORDER = ["C", "D"]
BANNED_TEXT = [
    "moral",
    "cooperation",
    "self-interest",
    "dictator game",
    "ultimatum game",
    "trust game",
    "belief",
    "expect",
]


def validate(data: pd.DataFrame) -> list[str]:
    required = [
        "vignette_id",
        "structure",
        "sender_category",
        "sender_action",
        "receiver_action",
        "setting",
        "relationship",
        "text",
    ]
    assert data.columns.tolist() == required
    assert len(data) == 60
    assert data["vignette_id"].nunique() == 60
    assert not data["text"].duplicated().any()
    assert set(data["structure"]) == set(STRUCTURE_ORDER)
    assert set(data["sender_category"]) == set(CATEGORY_ORDER)
    assert set(data["sender_action"]) == set(ACTION_ORDER)

    dg = data[data["structure"] == "DG"]
    strategic = data[data["structure"].isin(["UG", "TG"])]
    assert dg["receiver_action"].isna().all()
    assert strategic["receiver_action"].isin(ACTION_ORDER).all()

    cell_counts = data.groupby(
        ["structure", "sender_category", "sender_action", "receiver_action"],
        dropna=False,
    ).size()
    assert len(cell_counts) == 30
    assert (cell_counts == 2).all()
    assert (
        data.groupby(
            ["structure", "sender_category", "sender_action", "receiver_action"],
            dropna=False,
        )["setting"].nunique()
        == 2
    ).all()

    setting_counts = data["setting"].value_counts()
    assert setting_counts.to_dict() == {"individual": 30, "market": 30}
    assert data.loc[data["setting"] == "individual", "relationship"].value_counts().to_dict() == {
        "known": 15,
        "unfamiliar": 15,
    }
    assert data.loc[data["setting"] == "market", "relationship"].value_counts().to_dict() == {
        "ongoing": 15,
        "anonymous": 15,
    }

    word_counts = data["text"].str.split().str.len()
    assert word_counts.between(45, 80).all()
    for term in BANNED_TEXT:
        assert not data["text"].str.contains(term, case=False, regex=False).any(), term

    lines = [
        "JOINT-ACTION PILOT CHECKS",
        "",
        f"Rows: {len(data)}",
        f"Unique IDs: {data['vignette_id'].nunique()}",
        f"Conceptual cells: {len(cell_counts)}",
        "Vignettes per conceptual cell: 2",
        f"Individual / market: {setting_counts['individual']} / {setting_counts['market']}",
        "Individual known / unfamiliar: 15 / 15",
        "Market ongoing / anonymous: 15 / 15",
        f"Word-count range: {word_counts.min()}--{word_counts.max()}",
        f"Mean word count: {word_counts.mean():.1f}",
        "Exact duplicate texts: 0",
        "Banned analytical labels or expectation terms in text: 0",
        "DG receiver actions: all missing by design",
        "UG/TG receiver actions: all present",
        "",
        "Structure x setting:",
        pd.crosstab(data["structure"], data["setting"]).to_string(),
    ]
    return lines


def render(data: pd.DataFrame) -> str:
    lines = [
        "# Joint-action vignette pilot",
        "",
        "This is a researcher-readable rendering of `01_joint_action_pilot.csv`.",
        "Metadata identify the intended cells and should not be shown to the similarity rater.",
        "",
    ]
    for structure in STRUCTURE_ORDER:
        lines.extend([f"## {structure}-structured vignettes", ""])
        for category in CATEGORY_ORDER:
            for sender_action in ACTION_ORDER:
                receiver_actions = [None] if structure == "DG" else ACTION_ORDER
                for receiver_action in receiver_actions:
                    subset = data[
                        (data["structure"] == structure)
                        & (data["sender_category"] == category)
                        & (data["sender_action"] == sender_action)
                    ]
                    if receiver_action is not None:
                        subset = subset[subset["receiver_action"] == receiver_action]
                    receiver_label = "NA" if receiver_action is None else receiver_action
                    lines.extend(
                        [
                            f"### {category}; sender {sender_action}; receiver {receiver_label}",
                            "",
                        ]
                    )
                    for row in subset.sort_values("setting").itertuples(index=False):
                        lines.extend(
                            [
                                f"**{row.vignette_id} - {row.setting}, {row.relationship}.**",
                                "",
                                row.text,
                                "",
                            ]
                        )
    return "\n".join(lines)


def main() -> None:
    data = pd.read_csv(SOURCE)
    CHECKS.write_text("\n".join(validate(data)) + "\n", encoding="utf-8")
    READABLE.write_text(render(data), encoding="utf-8")
    print(f"Wrote {CHECKS.name} and {READABLE.name}")


if __name__ == "__main__":
    main()
