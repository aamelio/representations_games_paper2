#!/usr/bin/env python3
"""Extract the final vignettes and build the current blinded rating inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "rating_runs" / "dg_c_neutral"
VIGNETTE_SOURCE = ROOT / "input" / "joint_action_vignettes_final.docx"
CONTEXT_SOURCE = ROOT / "input" / "game_contexts.md"
SEED = 2026082109
GAME_ORDER = ["DG", "UG", "TG"]
FRAME_ORDER = ["Control", "Market", "Aid", "Bonus"]
GAME_CONTEXT_KEYS = {
    "DG": {"Control": "C-KW", "Market": "M-KW"},
    "UG": {"Control": "C-UG", "Market": "M-UG"},
    "TG": {"Control": "C-TG", "Market": "M-TG"},
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_vignettes() -> list[dict]:
    paragraphs = [paragraph.text.strip() for paragraph in Document(VIGNETTE_SOURCE).paragraphs]
    headings = [
        (index, match.group(1))
        for index, text in enumerate(paragraphs)
        if (match := re.fullmatch(r"## ((?:DG|UG|TG)-[A-Z-]+)", text))
    ]
    rows: list[dict] = []
    for position, (start, source_id) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(paragraphs)
        block = [text for text in paragraphs[start + 1 : end] if text]
        metadata: dict[str, str] = {}
        prose: list[str] = []
        for text in block:
            match = re.fullmatch(r"- \*\*([^*]+):\*\*\s*(.+)", text)
            if match:
                metadata[match.group(1).strip()] = match.group(2).strip()
            elif not text.startswith("#"):
                prose.append(text)
        rows.append(
            {
                "source_id": source_id,
                "game": source_id.split("-", 1)[0],
                "setting": metadata.get("Setting", ""),
                "sender_category": metadata.get("Sender category", ""),
                "sender_action": metadata.get("Sender action", ""),
                "belief": metadata.get("Belief", ""),
                "receiver_action": metadata.get("Receiver action", ""),
                "joint_action": metadata.get("Joint action", ""),
                "text": " ".join(prose),
            }
        )

    observed = {game: sum(row["game"] == game for row in rows) for game in GAME_ORDER}
    expected = {"DG": 6, "UG": 8, "TG": 8}
    if observed != expected:
        raise ValueError(f"Unexpected vignette counts: {observed}; expected {expected}")
    if len({row["source_id"] for row in rows}) != 22:
        raise ValueError("Vignette source identifiers are not unique")
    required = ["setting", "sender_category", "sender_action", "receiver_action", "joint_action", "text"]
    if any(not row[field] for row in rows for field in required):
        raise ValueError("At least one vignette is missing required metadata or prose")
    if {row["sender_category"] for row in rows} != {"Moral", "Self-interest", "Cooperation"}:
        raise ValueError("Unexpected sender-category labels")
    return rows


def clean_context_text(block: str) -> str:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", block.strip()):
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact or compact.startswith("[On the next screen"):
            continue
        if compact.startswith((
            "The allocation chosen will be implemented for",
            "The final allocation will be implemented for",
            "The final outcome will be implemented for",
        )):
            continue
        compact = re.sub(
            r"\s+Your decision affects a bonus payment\. Specifically, 1 in 10 participant "
            r"pairs will be randomly selected, and for those selected:$",
            "",
            compact,
        )
        paragraphs.append(compact)
    return "\n\n".join(paragraphs)


def parse_contexts() -> dict[str, str]:
    raw = CONTEXT_SOURCE.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## ([A-Z-]+) \([^\n]+\)\s*$", raw, flags=re.M))
    contexts = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        contexts[match.group(1)] = clean_context_text(raw[match.end() : end])
    needed = {"AID", "BONUS"}
    for game in GAME_ORDER:
        needed.update(GAME_CONTEXT_KEYS[game].values())
    missing = sorted(needed.difference(contexts))
    if missing:
        raise ValueError(f"Missing source contexts: {missing}")
    return contexts


def build_contexts(cleaned: dict[str, str]) -> list[dict]:
    rows = []
    for game in GAME_ORDER:
        control = cleaned[GAME_CONTEXT_KEYS[game]["Control"]]
        texts = {
            "Control": control,
            "Market": cleaned[GAME_CONTEXT_KEYS[game]["Market"]],
            "Aid": cleaned["AID"] + "\n\n" + control,
            "Bonus": cleaned["BONUS"] + "\n\n" + control,
        }
        for frame in FRAME_ORDER:
            rows.append({"source_id": f"{game}_{frame.upper()}", "game": game, "frame": frame, "text": texts[frame]})
    return rows


def anonymize(rows: list[dict], prefix: str, rng: random.Random) -> tuple[list[dict], dict]:
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    public_rows = []
    mapping = {}
    for index, row in enumerate(shuffled, start=1):
        neutral_id = f"{prefix}{index:02d}"
        public_rows.append({f"{prefix.lower()}_id": neutral_id, "text": row["text"]})
        mapping[neutral_id] = {key: value for key, value in row.items() if key != "text"}
    return public_rows, mapping


def main() -> None:
    vignettes = parse_vignettes()
    contexts = build_contexts(parse_contexts())
    rng = random.Random(SEED)
    public_vignettes, vignette_map = anonymize(vignettes, "V", rng)
    public_contexts, context_map = anonymize(contexts, "T", rng)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RUN_DIR / "anonymized_vignettes.csv", ["v_id", "text"], public_vignettes)
    write_csv(RUN_DIR / "anonymized_contexts.csv", ["t_id", "text"], public_contexts)
    write_csv(ROOT / "input" / "vignettes_catalog.csv", list(vignettes[0]), vignettes)
    mapping = {
        "seed": SEED,
        "source_hashes_sha256": {
            str(VIGNETTE_SOURCE.relative_to(ROOT)): hashlib.sha256(VIGNETTE_SOURCE.read_bytes()).hexdigest(),
            str(CONTEXT_SOURCE.relative_to(ROOT)): hashlib.sha256(CONTEXT_SOURCE.read_bytes()).hexdigest(),
        },
        "notes": {
            "rating_pool": "Every context is rated against all 22 DG-C-neutral vignettes in Stage 1.",
            "sensitivity_change": "Only DG-C-P and DG-C-K differ from round 8; incidental happiness and own-payoff wording was removed.",
            "within_game_choice_sets": {"DG": 6, "UG": 8, "TG": 8},
            "story_order": "Aid/Bonus story first, then cleaned abstract-game instructions.",
        },
        "vignettes": vignette_map,
        "contexts": context_map,
    }
    (RUN_DIR / "anonymization_map.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(public_vignettes)} revised vignettes and {len(public_contexts)} contexts")


if __name__ == "__main__":
    main()
