#!/usr/bin/env python3
"""Task 7 — cross-model agreement check and production batch scoring.

--agree            re-score the 48 pilot summaries with a cheaper model (live calls)
                   and report per-aspect agreement with the Opus 4.8 pilot scores.
--submit           build and submit ONE Message Batch scoring all summaries in
                   summaries_p1.csv with --model; writes batch_info.json.
--wait             poll the batch in batch_info.json; if it ends within --max-minutes,
                   collect results into production_scores.csv; else exit code 3
                   (relaunch to keep waiting). Results are keyed by custom_id
                   (s<row-index> into summaries_p1.csv), never by position.

The scoring prompt is imported from 02_extract_and_pilot.py so the instrument is
byte-identical across pilot, agreement check, and production.
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("pilot02", HERE / "02_extract_and_pilot.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

MAX_TOKENS = 2000


def call_model(model_id: str, text: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    with client.messages.stream(
        model=model_id, max_tokens=MAX_TOKENS,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": p2.PROMPT.format(summary=text)}],
    ) as stream:
        resp = stream.get_final_message()
    assert resp.stop_reason in ("end_turn", "stop_sequence"), resp.stop_reason
    reply = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", reply, flags=re.S)
    assert m, f"no JSON in reply: {reply[:200]}"
    return json.loads(m.group(0))


def agree(model_id: str) -> None:
    pilot = pd.read_csv(HERE / "pilot_scores.csv")
    summaries = pd.read_csv(HERE / "summaries_p1.csv")
    texts = summaries.set_index("PROLIFIC_PID")["summary"]
    rows = []
    for i, r in pilot.iterrows():
        s = call_model(model_id, texts[r["PROLIFIC_PID"]])
        rows.append(dict(PROLIFIC_PID=r["PROLIFIC_PID"],
                         F=s["F"], O=s["O"], J=s["J"], dominant=s["dominant"]))
        print(f"  {i + 1}/{len(pilot)} scored")
    alt = pd.DataFrame(rows)
    alt.to_csv(HERE / f"agreement_scores_{model_id}.csv", index=False)

    merged = pilot.merge(alt, on="PROLIFIC_PID", suffixes=("_opus", "_alt"))
    print(f"\n=== agreement: {model_id} vs claude-opus-4-8, N={len(merged)} ===")
    for a in ["F", "O", "J"]:
        x, y = merged[f"{a}_opus"], merged[f"{a}_alt"]
        print(f"  {a}: pearson r={x.corr(y):.3f}  mean abs diff={(x - y).abs().mean():.1f}  "
              f"mean opus={x.mean():.1f} alt={y.mean():.1f}")
    dom = (merged["dominant_opus"] == merged["dominant_alt"]).mean()
    print(f"  dominant category agreement: {dom:.0%}")
    print("\n  Aid-Bonus contrast per aspect (alt model):")
    m = merged.groupby("story")[["F_alt", "O_alt", "J_alt"]].mean()  # story is in pilot_scores
    for a in ["F", "O", "J"]:
        print(f"    {a}: aid {m.loc['aid', a + '_alt']:.1f}  bonus {m.loc['bonus', a + '_alt']:.1f}")


def submit(model_id: str) -> None:
    import anthropic
    client = anthropic.Anthropic()
    df = pd.read_csv(HERE / "summaries_p1.csv")
    requests = [
        {
            "custom_id": f"s{i:04d}",
            "params": {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "output_config": {"effort": "high"},
                "messages": [{"role": "user",
                              "content": p2.PROMPT.format(summary=r["summary"])}],
            },
        }
        for i, r in df.iterrows()
    ]
    batch = client.messages.batches.create(requests=requests)
    (HERE / "batch_info.json").write_text(json.dumps(
        {"batch_id": batch.id, "model": model_id, "n": len(requests)}, indent=2))
    print(f"submitted batch {batch.id}: {len(requests)} requests on {model_id}")
    print(f"status: {batch.processing_status}")


def wait(max_minutes: float) -> None:
    import anthropic
    client = anthropic.Anthropic()
    info = json.loads((HERE / "batch_info.json").read_text())
    deadline = time.monotonic() + max_minutes * 60
    while True:
        batch = client.messages.batches.retrieve(info["batch_id"])
        c = batch.request_counts
        print(f"status={batch.processing_status} processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}", flush=True)
        if batch.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            print("still processing at max-minutes; relaunch --wait to continue")
            sys.exit(3)
        time.sleep(30)

    df = pd.read_csv(HERE / "summaries_p1.csv")
    rows, errors = [], []
    for result in client.messages.batches.results(info["batch_id"]):
        idx = int(result.custom_id[1:])
        if result.result.type != "succeeded":
            errors.append((result.custom_id, result.result.type))
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            errors.append((result.custom_id, "no_json"))
            continue
        try:
            s = json.loads(m.group(0))
            r = df.iloc[idx]
            rows.append(dict(PROLIFIC_PID=r["PROLIFIC_PID"], game=r["game"],
                             story=r["story"], F=s["F"], O=s["O"], J=s["J"],
                             dominant=s["dominant"], note=s.get("note", "")))
        except (json.JSONDecodeError, KeyError) as e:
            errors.append((result.custom_id, f"parse:{e}"))
    out = pd.DataFrame(rows).sort_values("PROLIFIC_PID")
    out.to_csv(HERE / "production_scores.csv", index=False)
    print(f"\nwrote production_scores.csv: {len(out)} scored, {len(errors)} failures")
    if errors:
        (HERE / "batch_errors.json").write_text(json.dumps(errors, indent=2))
        print("failures logged to batch_errors.json (first 5):", errors[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-minutes", type=float, default=8.5)
    args = ap.parse_args()
    if not (args.agree or args.submit or args.wait):
        ap.error("pass --agree, --submit, and/or --wait")
    if args.agree:
        agree(args.model)
    if args.submit:
        submit(args.model)
    if args.wait:
        wait(args.max_minutes)
