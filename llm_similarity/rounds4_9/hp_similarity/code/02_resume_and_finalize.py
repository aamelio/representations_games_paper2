"""Resume, validate, track, and finalize the HP ratings.

The script is deliberately scan-based: a packet counts as complete only when a
full output file exists, has the expected identifiers in the expected order,
and contains integer ratings in the permitted range. This makes the workflow
safe to resume after a task or usage-limit interruption.
"""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
PACKETS = WORK / "packets"
LEGACY_OUTPUTS = WORK / "agent_outputs"
RATER_OUTPUTS = WORK / "similarity_rater_outputs"
CLASSIFICATION_RESUME = PACKETS / "classification_resume"
CLASSIFICATION_CHECKPOINTS = WORK / "classification_checkpoints"
PROGRESS = OUTPUT / "progress.json"

REFERENCES = {
    "A": {
        "packet_dir": PACKETS / "similarity_reference_A",
        "score_column": "similarity_reference_A",
        "substantive_name": "dg_kw_control",
    },
    "B": {
        "packet_dir": PACKETS / "similarity_reference_B",
        "score_column": "similarity_reference_B",
        "substantive_name": "dg_kw_market",
    },
}

CATEGORY_LABELS = {
    0: "No clear justification",
    1: "Moral",
    2: "Mutual Benefit / Cooperation",
    3: "Self-interest",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"hp_response_id": str})


def validate_output(
    output_path: Path,
    packet_path: Path,
    score_column: str,
    minimum: int,
    maximum: int,
) -> pd.DataFrame:
    output = read_csv(output_path)
    packet = read_csv(packet_path)
    expected_columns = ["hp_response_id", score_column]
    if list(output.columns) != expected_columns:
        raise ValueError(
            f"{output_path}: expected columns {expected_columns}, found {list(output.columns)}"
        )
    if output["hp_response_id"].tolist() != packet["hp_response_id"].tolist():
        raise ValueError(f"{output_path}: identifiers or row order do not match {packet_path}")
    numeric = pd.to_numeric(output[score_column], errors="coerce")
    if numeric.isna().any() or not (numeric == numeric.astype(int)).all():
        raise ValueError(f"{output_path}: {score_column} must contain integers only")
    if not numeric.between(minimum, maximum).all():
        raise ValueError(
            f"{output_path}: {score_column} must be between {minimum} and {maximum}"
        )
    output[score_column] = numeric.astype(int)
    return output


def copy_checkpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise ValueError(f"Existing checkpoint differs from recovered source: {target}")
        return
    shutil.copy2(source, target)


def initialize() -> None:
    """Preserve recovered work and create small classification checkpoints."""
    for rater in range(1, 4):
        for reference in REFERENCES:
            (RATER_OUTPUTS / f"rater_{rater}" / f"reference_{reference}").mkdir(
                parents=True, exist_ok=True
            )

    # Recovered A files form the completed A side of rater 1.
    for source in sorted(LEGACY_OUTPUTS.glob("similarity_A_*_rated.csv")):
        packet = REFERENCES["A"]["packet_dir"] / source.name.replace("_rated", "")
        validate_output(source, packet, REFERENCES["A"]["score_column"], 0, 100)
        copy_checkpoint(
            source,
            RATER_OUTPUTS / "rater_1" / "reference_A" / source.name,
        )

    # Recovered B files form the completed portion of rater 2's B side.
    for source in sorted(LEGACY_OUTPUTS.glob("similarity_B_*_rated.csv")):
        packet = REFERENCES["B"]["packet_dir"] / source.name.replace("_rated", "")
        validate_output(source, packet, REFERENCES["B"]["score_column"], 0, 100)
        copy_checkpoint(
            source,
            RATER_OUTPUTS / "rater_2" / "reference_B" / source.name,
        )

    CLASSIFICATION_RESUME.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    pending = read_csv(PACKETS / "classification" / "classification_09.csv")
    if len(pending) != 400:
        raise ValueError(f"Expected 400 rows in classification_09.csv; found {len(pending)}")
    for part_number, start in enumerate(range(0, len(pending), 50), start=1):
        part = pending.iloc[start : start + 50]
        path = CLASSIFICATION_RESUME / f"classification_09_part_{part_number:02d}.csv"
        if path.exists():
            existing = read_csv(path)
            if not existing.equals(part.reset_index(drop=True)):
                raise ValueError(f"Existing classification part differs: {path}")
        else:
            part.to_csv(path, index=False)

    recovery = {
        "design": "three independent similarity ratings per HP-response/reference pair",
        "aggregation": "arithmetic mean of the three integer ratings",
        "recovered_similarity_A_files_assigned_to": "rater_1",
        "recovered_similarity_B_files_assigned_to": "rater_2",
        "classification": {
            "inherited_hpmin": 1200,
            "recovered_gpt_5_6": 3200,
            "remaining_gpt_5_6": 400,
        },
        "blinding": (
            "Similarity raters receive only the frozen similarity prompt, one neutral "
            "reference, and blinded hp_response_id/memory packets."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "resume_design.json").write_text(
        json.dumps(recovery, indent=2) + "\n", encoding="utf-8"
    )
    write_progress()


def similarity_status() -> tuple[dict[str, object], list[str]]:
    status: dict[str, object] = {}
    issues: list[str] = []
    for rater in range(1, 4):
        rater_status: dict[str, object] = {}
        for reference, spec in REFERENCES.items():
            complete: list[int] = []
            for number in range(1, 13):
                packet = spec["packet_dir"] / f"similarity_{reference}_{number:02d}.csv"
                output = (
                    RATER_OUTPUTS
                    / f"rater_{rater}"
                    / f"reference_{reference}"
                    / f"similarity_{reference}_{number:02d}_rated.csv"
                )
                if not output.exists():
                    continue
                try:
                    validate_output(output, packet, spec["score_column"], 0, 100)
                    complete.append(number)
                except ValueError as exc:
                    issues.append(str(exc))
            rater_status[f"reference_{reference}"] = {
                "complete_packets": complete,
                "n_complete_packets": len(complete),
                "n_complete_rows": 400 * len(complete),
                "next_missing_packet": next(
                    (number for number in range(1, 13) if number not in complete), None
                ),
            }
        status[f"rater_{rater}"] = rater_status
    return status, issues


def classification_status() -> tuple[dict[str, object], list[str]]:
    complete_legacy: list[int] = []
    complete_parts: list[int] = []
    issues: list[str] = []
    for number in range(1, 9):
        packet = PACKETS / "classification" / f"classification_{number:02d}.csv"
        output = LEGACY_OUTPUTS / f"classification_{number:02d}_rated.csv"
        if not output.exists():
            continue
        try:
            validate_output(output, packet, "category_num", 0, 3)
            complete_legacy.append(number)
        except ValueError as exc:
            issues.append(str(exc))
    for part_number in range(1, 9):
        packet = CLASSIFICATION_RESUME / f"classification_09_part_{part_number:02d}.csv"
        output = (
            CLASSIFICATION_CHECKPOINTS
            / f"classification_09_part_{part_number:02d}_rated.csv"
        )
        if not output.exists():
            continue
        try:
            validate_output(output, packet, "category_num", 0, 3)
            complete_parts.append(part_number)
        except ValueError as exc:
            issues.append(str(exc))
    return {
        "complete_original_packets": complete_legacy,
        "complete_resume_parts": complete_parts,
        "n_new_classifications_complete": 400 * len(complete_legacy) + 50 * len(complete_parts),
        "n_new_classifications_required": 3600,
        "next_missing_resume_part": next(
            (number for number in range(1, 9) if number not in complete_parts), None
        ),
    }, issues


def write_progress() -> dict[str, object]:
    similarity, similarity_issues = similarity_status()
    classification, classification_issues = classification_status()
    progress = {
        "similarity": similarity,
        "classification": classification,
        "validation_issues": similarity_issues + classification_issues,
    }
    PROGRESS.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(progress, indent=2))
    return progress


def collect_similarity(rater: int, reference: str) -> pd.DataFrame:
    spec = REFERENCES[reference]
    pieces = []
    for number in range(1, 13):
        packet = spec["packet_dir"] / f"similarity_{reference}_{number:02d}.csv"
        output = (
            RATER_OUTPUTS
            / f"rater_{rater}"
            / f"reference_{reference}"
            / f"similarity_{reference}_{number:02d}_rated.csv"
        )
        if not output.exists():
            raise FileNotFoundError(output)
        pieces.append(validate_output(output, packet, spec["score_column"], 0, 100))
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined) != 4800 or combined["hp_response_id"].duplicated().any():
        raise ValueError(f"Unexpected combined similarity rows for rater {rater}, reference {reference}")
    return combined


def collect_classification() -> pd.DataFrame:
    pieces = []
    for number in range(1, 9):
        packet = PACKETS / "classification" / f"classification_{number:02d}.csv"
        output = LEGACY_OUTPUTS / f"classification_{number:02d}_rated.csv"
        if not output.exists():
            raise FileNotFoundError(output)
        pieces.append(validate_output(output, packet, "category_num", 0, 3))
    for part_number in range(1, 9):
        packet = CLASSIFICATION_RESUME / f"classification_09_part_{part_number:02d}.csv"
        output = (
            CLASSIFICATION_CHECKPOINTS
            / f"classification_09_part_{part_number:02d}_rated.csv"
        )
        if not output.exists():
            raise FileNotFoundError(output)
        pieces.append(validate_output(output, packet, "category_num", 0, 3))
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined) != 3600 or combined["hp_response_id"].duplicated().any():
        raise ValueError("Unexpected combined classification rows")
    return combined


def finalize() -> None:
    progress = write_progress()
    if progress["validation_issues"]:
        raise ValueError("Cannot finalize while validation issues remain")

    master = read_csv(WORK / "hp_master_working.csv")
    classification = collect_classification().rename(
        columns={"category_num": "new_category_num"}
    )
    final = master.merge(classification, on="hp_response_id", how="left", validate="one_to_one")
    inherited = pd.to_numeric(final["inherited_category_num"], errors="coerce")
    new = pd.to_numeric(final["new_category_num"], errors="coerce")
    final["category_num"] = inherited.fillna(new).astype(int)
    final["category"] = final["category_num"].map(CATEGORY_LABELS)
    final["classification_origin"] = inherited.notna().map(
        {True: "inherited_hpmin", False: "gpt_5_6_round7"}
    )
    if final["category"].isna().any() or final["new_category_num"].notna().sum() != 3600:
        raise ValueError("Classification merge is incomplete")

    agreement_rows: list[dict[str, object]] = []
    for reference, spec in REFERENCES.items():
        substantive = spec["substantive_name"]
        rating_columns = []
        for rater in range(1, 4):
            column = f"similarity_{substantive}_rater_{rater}"
            rating_columns.append(column)
            ratings = collect_similarity(rater, reference).rename(
                columns={spec["score_column"]: column}
            )
            final = final.merge(ratings, on="hp_response_id", how="left", validate="one_to_one")
        if final[rating_columns].isna().any().any():
            raise ValueError(f"Similarity merge is incomplete for reference {reference}")
        final[f"similarity_{substantive}_mean"] = final[rating_columns].mean(axis=1)
        for first, second in combinations(rating_columns, 2):
            agreement_rows.append(
                {
                    "reference": reference,
                    "substantive_reference": substantive,
                    "rater_a": first,
                    "rater_b": second,
                    "pearson": final[first].corr(final[second], method="pearson"),
                    "spearman": final[first].corr(final[second], method="spearman"),
                    "n": len(final),
                }
            )

    drop_columns = ["inherited_category_num", "inherited_category", "new_category_num"]
    final = final.drop(columns=drop_columns)
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    final.to_csv(DATA / "hp_responses_classified_and_rated.csv", index=False)
    final.to_excel(DATA / "hp_responses_classified_and_rated.xlsx", index=False)
    pd.DataFrame(agreement_rows).to_csv(
        OUTPUT / "similarity_rater_agreement.csv", index=False
    )
    summary = {
        "rows": len(final),
        "inherited_classifications": int(
            (final["classification_origin"] == "inherited_hpmin").sum()
        ),
        "new_gpt_5_6_classifications": int(
            (final["classification_origin"] == "gpt_5_6_round7").sum()
        ),
        "similarity_raters_per_reference": 3,
        "similarity_aggregation": "arithmetic mean",
        "outputs": [
            "data/hp_responses_classified_and_rated.csv",
            "data/hp_responses_classified_and_rated.xlsx",
            "output/similarity_rater_agreement.csv",
        ],
    }
    (OUTPUT / "final_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["initialize", "progress", "finalize"])
    args = parser.parse_args()
    if args.action == "initialize":
        initialize()
    elif args.action == "progress":
        write_progress()
    else:
        finalize()


if __name__ == "__main__":
    main()
