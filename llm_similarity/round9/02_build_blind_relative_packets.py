#!/usr/bin/env python3
"""Build separately shuffled, classification-free packets for three raters."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "relative_packets"
RATER_SEEDS = {1: 84559, 2: 92173, 3: 104729}


def main() -> None:
    contexts = pd.read_csv(HERE / "anonymized_contexts.csv").set_index("t_id")["text"]
    vignettes = pd.read_csv(HERE / "anonymized_vignettes.csv").set_index("v_id")["text"]
    private_map = json.loads((HERE / "anonymization_map.json").read_text(encoding="utf-8"))
    vignette_game = {key: value["game"] for key, value in private_map["vignettes"].items()}
    context_game = {key: value["game"] for key, value in private_map["contexts"].items()}
    all_ids = sorted(vignettes.index)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for rater, seed in RATER_SEEDS.items():
        rng = random.Random(seed)
        tasks = []
        task_number = 1
        for context_id in sorted(contexts.index):
            for stage in ["structural_all22", "within_game"]:
                candidate_ids = all_ids.copy() if stage == "structural_all22" else [
                    vignette_id for vignette_id in all_ids
                    if vignette_game[vignette_id] == context_game[context_id]
                ]
                rng.shuffle(candidate_ids)
                tasks.append({
                    "task_id": f"R{task_number:02d}",
                    "stage": stage,
                    "context_id": context_id,
                    "context_text": contexts.loc[context_id],
                    "vignettes": [
                        {"vignette_id": vignette_id, "text": vignettes.loc[vignette_id]}
                        for vignette_id in candidate_ids
                    ],
                })
                task_number += 1
        rng.shuffle(tasks)
        packet = {
            "rater": rater,
            "instructions_file": "../relative_rating_prompt.md",
            "important": (
                "Use only this packet and the instructions file. Do not open the anonymization map, "
                "source DOCX or extracted data, other raters' packets, existing ratings, analysis code, "
                "figures, or outputs."
            ),
            "tasks": tasks,
        }
        (OUTPUT_DIR / f"rater_{rater}_tasks.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Wrote three blinded packets with 24 tasks each to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
