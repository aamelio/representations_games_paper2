"""Prepare blinded HP classification and similarity packets.

The script never changes source workbooks. It creates a working master file,
copies the 1,200 existing HPmin classifications by an exact row match, and
writes neutral packets containing only a row id and the HP description.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
PROMPTS = ROOT / "prompts"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"
HP_ALL = INPUT / "hp_all.xlsx"
HPMIN = INPUT / "hpmin_classified.xlsx"

PACKETS = WORK / "packets"
CLASSIFICATION_PACKETS = PACKETS / "classification"
SIMILARITY_A_PACKETS = PACKETS / "similarity_reference_A"
SIMILARITY_B_PACKETS = PACKETS / "similarity_reference_B"

MATCH_KEYS = ["PROLIFIC_PID", "treatment", "hp", "memory"]
CATEGORY_CHUNK_SIZE = 400
SIMILARITY_CHUNK_SIZE = 400


def write_chunks(frame: pd.DataFrame, directory: Path, prefix: str, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob(f"{prefix}_*.csv"):
        old.unlink()
    for start in range(0, len(frame), size):
        chunk = frame.iloc[start : start + size]
        number = start // size + 1
        chunk.to_csv(directory / f"{prefix}_{number:02d}.csv", index=False)


def recover_classification_prompt() -> str:
    return (PROMPTS / "classification_prompt.txt").read_text(encoding="utf-8").strip() + "\n"


def recover_dg_contexts() -> dict[str, str]:
    return {
        "Control": (INPUT / "reference_control.txt").read_text(encoding="utf-8").strip(),
        "Market": (INPUT / "reference_market.txt").read_text(encoding="utf-8").strip(),
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hp = pd.read_excel(HP_ALL).reset_index(drop=True)
    hpmin = pd.read_excel(HPMIN)

    if len(hp) != 4800 or len(hpmin) != 1200:
        raise ValueError(f"Unexpected source sizes: all={len(hp)}, hpmin={len(hpmin)}")
    if hp.duplicated(MATCH_KEYS).any() or hpmin.duplicated(MATCH_KEYS).any():
        raise ValueError("Exact HP matching keys are not unique")

    inherited = hpmin[MATCH_KEYS + ["category_num", "category"]].copy()
    inherited = inherited.rename(
        columns={"category_num": "inherited_category_num", "category": "inherited_category"}
    )
    master = hp.merge(inherited, on=MATCH_KEYS, how="left", validate="one_to_one")
    master.insert(0, "hp_response_id", [f"HP{i:04d}" for i in range(1, len(master) + 1)])
    master.insert(1, "source_row", range(1, len(master) + 1))
    master["category_num"] = master["inherited_category_num"]
    master["category"] = master["inherited_category"]
    master["classification_origin"] = master["category_num"].notna().map(
        {True: "inherited_hpmin", False: "pending_gpt_5_6"}
    )

    inherited_n = int(master["category_num"].notna().sum())
    if inherited_n != 1200:
        raise ValueError(f"Expected 1,200 inherited labels; found {inherited_n}")

    master.to_csv(WORK / "hp_master_working.csv", index=False)

    blind_columns = ["hp_response_id", "memory"]
    classification = master.loc[master["category_num"].isna(), blind_columns]
    similarity = master[blind_columns]
    if len(classification) != 3600 or len(similarity) != 4800:
        raise ValueError("Unexpected blinded packet sizes")

    write_chunks(
        classification,
        CLASSIFICATION_PACKETS,
        "classification",
        CATEGORY_CHUNK_SIZE,
    )
    write_chunks(similarity, SIMILARITY_A_PACKETS, "similarity_A", SIMILARITY_CHUNK_SIZE)
    write_chunks(similarity, SIMILARITY_B_PACKETS, "similarity_B", SIMILARITY_CHUNK_SIZE)

    contexts = recover_dg_contexts()
    (OUTPUT / "private_reference_map.json").write_text(
        json.dumps({"A": "DG-KW Control", "B": "DG-KW Market"}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "all_hp_rows": len(master),
        "inherited_hpmin_labels": inherited_n,
        "new_classifications_required": len(classification),
        "similarity_ratings_per_reference": len(similarity),
        "classification_chunks": len(list(CLASSIFICATION_PACKETS.glob("*.csv"))),
        "similarity_chunks_per_reference": len(list(SIMILARITY_A_PACKETS.glob("*.csv"))),
        "matching_keys": MATCH_KEYS,
        "category_chunk_size": CATEGORY_CHUNK_SIZE,
        "similarity_chunk_size": SIMILARITY_CHUNK_SIZE,
    }
    (OUTPUT / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
