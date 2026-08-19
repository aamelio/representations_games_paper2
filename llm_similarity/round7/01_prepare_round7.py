"""Prepare blinded HP classification and similarity packets for round 7.

The script never changes source workbooks. It creates a working master file,
copies the 1,200 existing HPmin classifications by an exact row match, and
writes neutral packets containing only a row id and the HP description.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
DATA = PAPER / "replication_package" / "data"
HP_ALL = DATA / "hp_social_proximity_all.xlsx"
HPMIN = DATA / "hpmin_new_scheme_categorized.xlsx"
CLASSIFIER_CODE = PAPER / "replication_package" / "code" / "14_hp_classification.py"
ROUND6 = HERE.parent / "round6"

PACKETS = HERE / "packets"
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
    spec = importlib.util.spec_from_file_location("hp_classification_source", CLASSIFIER_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CLASSIFIER_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.HP_SYSTEM).strip() + "\n"


def recover_dg_contexts() -> dict[str, str]:
    mapping = json.loads((ROUND6 / "anonymization_map.json").read_text(encoding="utf-8"))
    contexts = pd.read_csv(ROUND6 / "anonymized_contexts.csv").set_index("t_id")
    found: dict[str, str] = {}
    for neutral_id, metadata in mapping["contexts"].items():
        if metadata["game"] == "DG" and metadata["frame"] in {"Control", "Market"}:
            found[metadata["frame"]] = str(contexts.at[neutral_id, "text"])
    if set(found) != {"Control", "Market"}:
        raise ValueError(f"Could not recover both DG contexts: {sorted(found)}")
    return found


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
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

    master.to_csv(HERE / "hp_master_working.csv", index=False)

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

    (HERE / "classification_prompt_used.txt").write_text(
        recover_classification_prompt(), encoding="utf-8"
    )
    contexts = recover_dg_contexts()
    (HERE / "reference_A.txt").write_text(contexts["Control"].strip() + "\n", encoding="utf-8")
    (HERE / "reference_B.txt").write_text(contexts["Market"].strip() + "\n", encoding="utf-8")
    (HERE / "private_reference_map.json").write_text(
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
    (HERE / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
