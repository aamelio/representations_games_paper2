#!/usr/bin/env python3
"""Validate and render the reduced 28-vignette joint-action pool.

Input:  01_reduced_joint_action_vignettes.csv
Outputs: reduced_joint_action_vignettes_readable.md,
         reduced_joint_action_vignettes_checks.txt
"""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "01_reduced_joint_action_vignettes.csv"
READABLE = HERE / "reduced_joint_action_vignettes_readable.md"
CHECKS = HERE / "reduced_joint_action_vignettes_checks.txt"

STRUCTURE_ORDER = ["DG", "UG", "TG"]
PROFILE_ORDER = ["Moral", "Self-interest", "Cooperation"]
RECEIVER_ORDER = {
    "DG": [None],
    "UG": ["accept", "reject"],
    "TG": ["return", "keep"],
}
BANNED_TEXT = [
    "moral",
    "self-interest",
    "cooperation",
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
        "sender_profile",
        "sender_keeps",
        "sender_gives",
        "receiver_action",
        "receiver_transfer",
        "setting",
        "market_anonymous",
        "text",
    ]
    assert data.columns.tolist() == required
    assert len(data) == 28
    assert data["vignette_id"].nunique() == 28
    assert not data["text"].duplicated().any()
    assert data["setting"].value_counts().to_dict() == {
        "personal": 14,
        "market": 14,
    }

    market_flags = data["market_anonymous"].astype(str).str.lower()
    assert (market_flags[data["setting"] == "market"] == "true").all()
    assert (market_flags[data["setting"] == "personal"] == "false").all()
    assert data.loc[data["setting"] == "market", "text"].str.contains(
        "anonymous", case=False, regex=False
    ).all()

    dg = data[data["structure"] == "DG"]
    ug = data[data["structure"] == "UG"]
    tg = data[data["structure"] == "TG"]
    assert len(dg) == 4 and len(ug) == 12 and len(tg) == 12
    assert set(dg["sender_profile"]) == {"Moral", "Self-interest"}
    assert set(ug["sender_profile"]) == set(PROFILE_ORDER)
    assert set(tg["sender_profile"]) == set(PROFILE_ORDER)
    assert dg["receiver_action"].isna().all()
    assert set(ug["receiver_action"]) == {"accept", "reject"}
    assert set(tg["receiver_action"]) == {"return", "keep"}

    retained_profiles = data[["structure", "sender_profile"]].drop_duplicates()
    for row in retained_profiles.itertuples(index=False):
        profile = data[
            (data["structure"] == row.structure)
            & (data["sender_profile"] == row.sender_profile)
        ]
        assert set(profile["setting"]) == {"personal", "market"}
        if row.structure in ["UG", "TG"]:
            assert set(profile["receiver_action"]) == set(RECEIVER_ORDER[row.structure])
            assert len(profile) == 4
        else:
            assert len(profile) == 2

    assert set(zip(dg["sender_keeps"], dg["sender_gives"])) == {(6, 6), (12, 0)}
    assert set(zip(ug["sender_keeps"], ug["sender_gives"])) == {(6, 6), (8, 4), (4, 8)}
    assert set(zip(tg["sender_keeps"], tg["sender_gives"])) == {(3, 3), (5, 1), (0, 6)}

    word_counts = data["text"].str.split().str.len()
    assert word_counts.between(40, 90).all()
    for term in BANNED_TEXT:
        assert not data["text"].str.contains(term, case=False, regex=False).any(), term

    return [
        "REDUCED JOINT-ACTION VIGNETTE CHECKS",
        "",
        f"Rows: {len(data)}",
        f"Unique IDs: {data['vignette_id'].nunique()}",
        "DG / UG / TG: 4 / 12 / 12",
        "Personal / anonymous market: 14 / 14",
        f"Word-count range: {word_counts.min()}--{word_counts.max()}",
        f"Mean word count: {word_counts.mean():.1f}",
        "Exact duplicate texts: 0",
        "Banned analytical labels or expectation terms in text: 0",
        "All market vignettes explicitly identify an anonymous interaction",
        "All retained UG/TG sender profiles have both receiver responses",
        "DG allocations: 6/6 and 12/0",
        "UG allocations: 6/6, 8/4, and 4/8",
        "TG transfers: 3, 1, and 6 out of 6",
        "",
        "Structure x setting:",
        pd.crosstab(data["structure"], data["setting"]).to_string(),
    ]


def render(data: pd.DataFrame) -> str:
    lines = [
        "# Reduced joint-action vignette pool",
        "",
        "This is a researcher-readable rendering of `01_reduced_joint_action_vignettes.csv`.",
        "The metadata headings identify intended cells and should not be shown to raters or survey respondents.",
        "",
    ]
    for structure in STRUCTURE_ORDER:
        lines.extend([f"## {structure}-structured vignettes", ""])
        for profile in PROFILE_ORDER:
            profile_data = data[
                (data["structure"] == structure)
                & (data["sender_profile"] == profile)
            ]
            if profile_data.empty:
                continue
            for receiver_action in RECEIVER_ORDER[structure]:
                subset = profile_data
                if receiver_action is not None:
                    subset = subset[subset["receiver_action"] == receiver_action]
                receiver_label = "no receiver action" if receiver_action is None else receiver_action
                allocation = (
                    f"sender keeps {int(subset.iloc[0]['sender_keeps'])}, "
                    f"sender gives {int(subset.iloc[0]['sender_gives'])}"
                )
                lines.extend(
                    [
                        f"### {profile}; {allocation}; receiver: {receiver_label}",
                        "",
                    ]
                )
                for row in subset.sort_values("setting", ascending=False).itertuples(index=False):
                    lines.extend(
                        [
                            f"**{row.vignette_id} - {row.setting}.**",
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
