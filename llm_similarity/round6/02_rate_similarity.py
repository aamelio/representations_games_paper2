#!/usr/bin/env python3
"""Rate all 12 blinded contexts against all 30 blinded vignettes.

Each context-replicate is a fresh API call.  Vignette presentation order is
randomized independently across replicates, while the saved output reconnects
the displayed labels to the stable neutral identifiers from the input CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_SEED = 20260817
PROMPT_VERSION = "round6-neutral-v1"
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
INPUT_FILES = [
    HERE / "anonymized_vignettes.csv",
    HERE / "anonymized_contexts.csv",
    HERE / "anonymization_map.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def input_fingerprint() -> str:
    digest = hashlib.sha256(PROMPT_VERSION.encode("utf-8"))
    for path in INPUT_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_prompt(context_text: str, vignettes: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    display_map: dict[str, str] = {}
    vignette_lines: list[str] = []
    for position, vignette in enumerate(vignettes, start=1):
        label = f"V{position:02d}"
        display_map[label] = vignette["v_id"]
        vignette_lines.append(f"{label}: {vignette['text']}")
    requested = ", ".join(f'"{label}": <0-100>' for label in display_map)
    prompt = f"""I am running a research project on how people perceive similarity between situations.

Below is one focal situation and a set of short vignettes. For each vignette, rate on a 0-100 scale how similar the focal situation is to the situation in that vignette.

0 = the two situations have no meaningful similarity
100 = the two situations are essentially the same situation

Assess every pair independently. Use your own judgment to decide which aspects are relevant and how much weight to give them. Apply the same standard to all vignettes. Equal ratings are allowed, and the ratings do not need to sum to any particular total.

FOCAL SITUATION

{context_text}

VIGNETTES

{chr(10).join(vignette_lines)}

Return a single JSON object and no other text, using exactly this structure:
{{"ratings": {{{requested}}}}}
"""
    return prompt, display_map


def extract_ratings(reply: str, labels: list[str]) -> dict[str, float]:
    match = re.search(r"\{.*\}", reply, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in response")
    data = json.loads(match.group(0))
    ratings = data.get("ratings")
    if not isinstance(ratings, dict):
        raise ValueError("Response has no ratings object")
    if set(ratings) != set(labels):
        missing = sorted(set(labels).difference(ratings))
        extra = sorted(set(ratings).difference(labels))
        raise ValueError(f"Rating keys mismatch; missing={missing}, extra={extra}")
    parsed = {label: float(ratings[label]) for label in labels}
    bad = {label: value for label, value in parsed.items() if not 0 <= value <= 100}
    if bad:
        raise ValueError(f"Ratings outside [0, 100]: {bad}")
    return parsed


def call_model(provider: str, model: str, prompt: str) -> str:
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(max_retries=8)
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    if provider == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    raise ValueError(f"Unsupported provider: {provider}")


def completed_units(output_path: Path, fingerprint: str) -> set[tuple[str, str, int, str]]:
    if not output_path.exists():
        return set()
    rows = read_csv(output_path)
    fingerprints = {row.get("input_fingerprint", "") for row in rows}
    if fingerprints != {fingerprint}:
        raise ValueError(
            "Existing ratings were produced from different or unversioned inputs. "
            "Use a separate output file or rerun explicitly with --overwrite."
        )
    counts: dict[tuple[str, str, int, str], int] = {}
    for row in rows:
        key = (row["provider"], row["model"], int(row["replicate"]), row["context_id"])
        counts[key] = counts.get(key, 0) + 1
    incomplete = {key: count for key, count in counts.items() if count != 30}
    if incomplete:
        raise ValueError(f"Ratings file contains incomplete or duplicated units: {incomplete}")
    return {key for key, count in counts.items() if count == 30}


def append_rows_atomic(path: Path, rows: list[dict]) -> None:
    fields = [
        "provider",
        "model",
        "replicate",
        "context_id",
        "vignette_id",
        "presentation_label",
        "presentation_order",
        "rating",
        "prompt_version",
        "attempts",
        "timestamp_utc",
        "input_fingerprint",
    ]
    existing = read_csv(path) if path.exists() else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)
    temporary.replace(path)


def write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(KEYS), default="anthropic")
    parser.add_argument("--model", help="Exact API model identifier")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contexts = read_csv(HERE / "anonymized_contexts.csv")
    vignettes = read_csv(HERE / "anonymized_vignettes.csv")
    if len(contexts) != 12 or len(vignettes) != 30:
        raise ValueError(f"Expected 12 contexts and 30 vignettes; got {len(contexts)} and {len(vignettes)}")

    preview_dir = HERE / "prompt_previews"
    if args.dry_run:
        preview_dir.mkdir(exist_ok=True)
        for replicate in range(1, args.replicates + 1):
            context_order = contexts.copy()
            random.Random(BASE_SEED + 100 * replicate).shuffle(context_order)
            for context in context_order:
                ordered = vignettes.copy()
                random.Random(BASE_SEED + 1000 * replicate + int(context["t_id"][1:])).shuffle(ordered)
                prompt, _ = build_prompt(context["text"], ordered)
                (preview_dir / f"rep{replicate}_{context['t_id']}.txt").write_text(prompt, encoding="utf-8")
        print(f"Dry run wrote {args.replicates * len(contexts)} prompt previews")
        return

    if not args.model:
        raise SystemExit("--model is required unless --dry-run is used")
    key_name = KEYS[args.provider]
    if not os.environ.get(key_name):
        raise SystemExit(f"{key_name} is not set; no paid API calls were made")

    fingerprint = input_fingerprint()
    output_path = HERE / "similarity_ratings.csv"
    if args.overwrite and output_path.exists():
        output_path.unlink()
    done = completed_units(output_path, fingerprint)
    transcript_dir = HERE / "transcripts"
    transcript_dir.mkdir(exist_ok=True)

    for replicate in range(1, args.replicates + 1):
        context_order = contexts.copy()
        random.Random(BASE_SEED + 100 * replicate).shuffle(context_order)
        for context in context_order:
            unit = (args.provider, args.model, replicate, context["t_id"])
            if unit in done:
                print(f"[skip] complete: replicate {replicate}, {context['t_id']}")
                continue
            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
            transcript_path = transcript_dir / f"{args.provider}_{safe_model}_rep{replicate}_{context['t_id']}.json"
            if transcript_path.exists() and not args.overwrite:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                if transcript.get("input_fingerprint") != fingerprint:
                    raise ValueError(f"Stale transcript has a different input fingerprint: {transcript_path}")
                prompt = transcript["prompt"]
                reply = transcript["response"]
                display_map = transcript["display_map"]
                attempt = int(transcript.get("attempts", 1))
                ratings = extract_ratings(reply, list(display_map))
                print(f"[recover] using saved transcript for replicate {replicate}, {context['t_id']}")
            else:
                ordered = vignettes.copy()
                random.Random(BASE_SEED + 1000 * replicate + int(context["t_id"][1:])).shuffle(ordered)
                prompt, display_map = build_prompt(context["text"], ordered)
                error: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        reply = call_model(args.provider, args.model, prompt)
                        ratings = extract_ratings(reply, list(display_map))
                        break
                    except Exception as exc:  # preserve the failed response through the final exception
                        error = exc
                        if attempt == 3:
                            raise
                        time.sleep(2**attempt)
                else:  # pragma: no cover
                    raise RuntimeError(error)

                transcript = {
                    "provider": args.provider,
                    "model": args.model,
                    "replicate": replicate,
                    "context_id": context["t_id"],
                    "prompt": prompt,
                    "response": reply,
                    "display_map": display_map,
                    "prompt_version": PROMPT_VERSION,
                    "attempts": attempt,
                    "input_fingerprint": fingerprint,
                }
                write_json_atomic(transcript_path, transcript)

            rows = []
            for order, label in enumerate(display_map, start=1):
                rows.append(
                    {
                        "provider": args.provider,
                        "model": args.model,
                        "replicate": replicate,
                        "context_id": context["t_id"],
                        "vignette_id": display_map[label],
                        "presentation_label": label,
                        "presentation_order": order,
                        "rating": ratings[label],
                        "prompt_version": PROMPT_VERSION,
                        "attempts": attempt,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "input_fingerprint": fingerprint,
                    }
                )
            append_rows_atomic(output_path, rows)
            print(f"[done] replicate {replicate}, {context['t_id']}")


if __name__ == "__main__":
    main()
