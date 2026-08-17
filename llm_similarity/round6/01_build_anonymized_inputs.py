#!/usr/bin/env python3
"""Build the blinded round-6 vignette and game-by-frame input files.

The source classifications are retained only in ``anonymization_map.json``.  The
two CSVs contain neutral identifiers and the text that may be shown to a rater.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
VIGNETTE_SOURCE = HERE / "joint_action_vignettes_classified.txt"
CONTEXT_SOURCE = HERE.parent / "round2" / "contexts.md"
SEED = 20260817

GAME_CONTEXT_KEYS = {
    "DG": {"Control": "C-KW", "Market": "M-KW"},
    "UG": {"Control": "C-UG", "Market": "M-UG"},
    "TG": {"Control": "C-TG", "Market": "M-TG"},
}
GAME_ORDER = ["DG", "UG", "TG"]
FRAME_ORDER = ["Control", "Market", "Aid", "Bonus"]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_vignettes() -> list[dict]:
    raw = VIGNETTE_SOURCE.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## ([A-Z0-9-]+)\s*$", raw, flags=re.M))
    rows: list[dict] = []

    for i, match in enumerate(headings):
        source_id = match.group(1)
        if not re.fullmatch(r"(?:DG|UG|TG)-[A-Z-]+", source_id):
            continue
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        block = raw[match.end() : end]
        metadata = {
            key.strip(): value.strip()
            for key, value in re.findall(r"^\* ([^:]+):\s*(.+?)\s*$", block, flags=re.M)
        }
        text_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("*") and not line.startswith("#")
        ]
        text = " ".join(text_lines)
        game = source_id.split("-", 1)[0]
        setting_code = source_id.rsplit("-", 1)[1]
        row = {
            "source_id": source_id,
            "game": game,
            "setting": metadata.get("Setting", ""),
            "setting_code": setting_code,
            "sender_category": metadata.get("Sender category", ""),
            "sender_action": metadata.get("Sender action", ""),
            "receiver_action": metadata.get("Receiver action", ""),
            "joint_action": metadata.get("Joint action", ""),
            "text": text,
        }
        rows.append(row)

    expected = {"DG": 6, "UG": 12, "TG": 12}
    observed = {game: sum(row["game"] == game for row in rows) for game in GAME_ORDER}
    if observed != expected:
        raise ValueError(f"Unexpected vignette counts: {observed}; expected {expected}")
    if any(not row["text"] for row in rows):
        raise ValueError("At least one vignette has no prose text")
    if len({row["source_id"] for row in rows}) != 30:
        raise ValueError("Vignette source identifiers are not unique")
    return rows


def parse_round2_contexts() -> dict[str, str]:
    raw = CONTEXT_SOURCE.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## ([A-Z-]+) \([^\n]+\)\s*$", raw, flags=re.M))
    contexts: dict[str, str] = {}
    for i, match in enumerate(headings):
        key = match.group(1)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        contexts[key] = _clean_context_text(raw[match.end() : end])

    needed = {"AID", "BONUS"}
    for game in GAME_ORDER:
        needed.update(GAME_CONTEXT_KEYS[game].values())
    missing = sorted(needed.difference(contexts))
    if missing:
        raise ValueError(f"Missing context sections in {CONTEXT_SOURCE}: {missing}")
    return contexts


def _clean_context_text(block: str) -> str:
    """Remove survey logistics while keeping payoff and decision content."""
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", block.strip()):
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact:
            continue
        if compact.startswith("[On the next screen"):
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


def build_contexts(cleaned: dict[str, str]) -> list[dict]:
    contexts: list[dict] = []
    for game in GAME_ORDER:
        control_text = cleaned[GAME_CONTEXT_KEYS[game]["Control"]]
        market_text = cleaned[GAME_CONTEXT_KEYS[game]["Market"]]
        texts = {
            "Control": control_text,
            "Market": market_text,
            # Experimental order: the story is read before the abstract game.
            "Aid": cleaned["AID"] + "\n\n" + control_text,
            "Bonus": cleaned["BONUS"] + "\n\n" + control_text,
        }
        for frame in FRAME_ORDER:
            contexts.append(
                {
                    "source_id": f"{game}_{frame.upper()}",
                    "game": game,
                    "frame": frame,
                    "text": texts[frame],
                }
            )
    return contexts


def anonymize(rows: list[dict], prefix: str, rng: random.Random) -> tuple[list[dict], dict]:
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    public_rows: list[dict] = []
    mapping: dict[str, dict] = {}
    width = 2
    for index, row in enumerate(shuffled, start=1):
        neutral_id = f"{prefix}{index:0{width}d}"
        public_rows.append({f"{prefix.lower()}_id": neutral_id, "text": row["text"]})
        mapping[neutral_id] = {key: value for key, value in row.items() if key != "text"}
    return public_rows, mapping


def main() -> None:
    vignettes = parse_vignettes()
    contexts = build_contexts(parse_round2_contexts())
    rng = random.Random(SEED)
    public_vignettes, vignette_map = anonymize(vignettes, "V", rng)
    public_contexts, context_map = anonymize(contexts, "T", rng)

    _write_csv(HERE / "anonymized_vignettes.csv", ["v_id", "text"], public_vignettes)
    _write_csv(HERE / "anonymized_contexts.csv", ["t_id", "text"], public_contexts)
    mapping = {
        "seed": SEED,
        "source_hashes_sha256": {
            VIGNETTE_SOURCE.name: hashlib.sha256(VIGNETTE_SOURCE.read_bytes()).hexdigest(),
            str(CONTEXT_SOURCE.relative_to(HERE.parent.parent)): hashlib.sha256(
                CONTEXT_SOURCE.read_bytes()
            ).hexdigest(),
        },
        "notes": {
            "rating_pool": "Every context is rated against all 30 vignettes.",
            "story_order": "Aid/Bonus story first, then cleaned abstract-game instructions.",
            "primary_plot_normalization": (
                "DG contexts use the 6 DG vignettes; UG and TG contexts use the combined 24 UG+TG vignettes."
            ),
        },
        "vignettes": vignette_map,
        "contexts": context_map,
    }
    (HERE / "anonymization_map.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(public_vignettes)} vignettes and {len(public_contexts)} contexts")


if __name__ == "__main__":
    main()
