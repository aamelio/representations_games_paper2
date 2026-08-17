"""Forced classification of the residual ("No clear justification") Player 1 answers,
Control and Market conditions (AA's example-based method, 2026-07 email package).

Method (AA): 95 residual answers were forced-classified into the three substantive
categories by both research assistants and by the earlier LLM pass; where the LLM label
agreed with at least one RA, the case becomes a labelled reference example. The remaining
214 residual answers are then classified in-context: the 95 labelled examples are placed
in the prompt and the model outputs a category for the new answer. AA's system prompt,
reference-block format, and query format are kept VERBATIM from his package
(02_classify_remaining_player1_with_examples.py); this port only fixes identifiers
(PROLIFIC_PID/game/story attached from the source workbook via source_row, because the
RA sheets lack them for some rows) and drops `temperature` for models that reject it.

Inputs (data/):
  classification_no_clear_justification.xlsx        RA 1 forced labels (cat_manual)
  control_no_clear_justification_reasons.xlsx       RA 2 forced labels (forced classification)
  player1_control_market_no_clear_reclassified.xlsx all 309 residual answers + earlier
                                                    forced LLM pass (reclassification_code)

Phases (run in order; --classify and --loo need ANTHROPIC_API_KEY):
  python3 18_unclassified_classification.py --build
  python3 18_unclassified_classification.py --classify --model claude-opus-4-8
  python3 18_unclassified_classification.py --classify --model claude-sonnet-4-6
  python3 18_unclassified_classification.py --loo --model claude-opus-4-8
  python3 18_unclassified_classification.py --loo --model claude-sonnet-4-6

Headline model: claude-opus-4-8, as for the re-classification in Appendix app:hp;
claude-sonnet-4-6 (AA's original choice) enters as a cross-model check. Runs of
2026-07-20. Both --classify and --loo checkpoint and resume from existing output.

Outputs (output/unclassified/):
  player1_reference_examples.csv        the 95 labelled examples
  classified_<model>.xlsx               forced categories for the 214 remaining answers
  loo_validation_<model>.csv            leave-one-out predictions for the 95
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "output" / "unclassified"

RA1_FILE = DATA / "classification_no_clear_justification.xlsx"
RA2_FILE = DATA / "control_no_clear_justification_reasons.xlsx"
LLM_FILE = DATA / "player1_control_market_no_clear_reclassified.xlsx"
REFERENCE_CSV = OUT / "player1_reference_examples.csv"

EXPECTED_REFERENCE_ROWS = 95
MAX_RETRIES = 6
CATEGORY_MAP = {1: "Moral", 2: "Mutual Benefit / Cooperation", 3: "Self-interest"}
VALID_CODES = {"1", "2", "3"}

SHEET_SPECS = [
    {"ra1_sheet": "player1_control", "ra1_code": "cat_manual",
     "ra2_sheet": "player1_control", "ra2_code": "forced classification",
     "key_columns": ["PROLIFIC_PID", "reasons"]},
    {"ra1_sheet": "player1_treated", "ra1_code": "cat_manual",
     "ra2_sheet": "Player 1_market", "ra2_code": "forced classification",
     "key_columns": ["game", "reasons"]},
]

# --- AA's prompt, verbatim from his package ---------------------------------------
SYSTEM_PROMPT = (
    "You classify Player 1 free-text survey responses from Dictator, Trust, or "
    "Ultimatum Games. A labelled reference set is provided in the user message. "
    "Use both the definitions and examples as guidance, but decide from the "
    "stated logic of the new response.\n\n"
    "1 = Moral: fairness, kindness, sharing, generosity, equality, or doing what "
    "is right.\n\n"
    "2 = Mutual benefit / productive partnership reasoning: creating value through "
    "cooperation, higher joint returns, investment, exchange, profitability, or both "
    "players benefiting. Trust-Game investment logic belongs here. Risk, uncertainty, "
    "hedging, or a safety net are category 2 when the main focus is a larger joint "
    "surplus or better overall outcome.\n\n"
    "3 = Strategic self-protection / self-interest reasoning: protecting, securing, "
    "or maximizing one's own outcome; keeping more; avoiding personal loss; "
    "mistrusting the other player; or avoiding exploitation.\n\n"
    "Classify only what is explicitly stated. There is no 'no clear justification' "
    "category. If multiple ideas appear, choose the main reason. Tie-breakers: "
    "profit, returns, upside, value creation, multiplier, both benefit, better "
    "overall, hedging, or safety net -> 2; betrayal, exploitation, distrust, "
    "protecting one's own payoff, or keeping more -> 3; fairness, equality, "
    "generosity, or decency -> 1.\n\n"
    "Output only one number: 1, 2, or 3."
)


# --- phase 1: build the 95 reference examples (AA's script 01, identifier fix) ------
def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", value).strip().casefold()


def add_merge_keys(db: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    keyed = db.copy()
    normalized = []
    for column in key_columns:
        nc = f"_{column}_key"
        keyed[nc] = keyed[column].map(normalize_text)
        normalized.append(nc)
    keyed["_occurrence"] = keyed.groupby(normalized, dropna=False).cumcount()
    return keyed


def load_ra_sheet(path: Path, sheet: str, code_column: str) -> pd.DataFrame:
    db = pd.read_excel(path, sheet_name=sheet)
    if "reason" in db.columns:
        db = db.rename(columns={"reason": "reasons"})
    missing = {"reasons", code_column}.difference(db.columns)
    if missing:
        raise KeyError(f"{path.name}, sheet '{sheet}' is missing {sorted(missing)}")
    db = db.copy()
    db["ra_code"] = pd.to_numeric(db[code_column], errors="coerce").astype("Int64")
    return db


def load_llm_workbook() -> pd.DataFrame:
    db = pd.read_excel(LLM_FILE)
    required = {"source_row", "PROLIFIC_PID", "game", "story", "reasons",
                "reclassification_code"}
    missing = required.difference(db.columns)
    if missing:
        raise KeyError(f"{LLM_FILE.name} is missing {sorted(missing)}")
    db = db.copy()
    db["llm_code"] = pd.to_numeric(db["reclassification_code"], errors="coerce").astype("Int64")
    if not db["source_row"].is_unique:
        raise ValueError("source_row must be unique in the LLM workbook.")
    return db


def build_reference() -> None:
    llm_rows = load_llm_workbook()
    selected = []
    for spec in SHEET_SPECS:
        keys = spec["key_columns"]
        merge_cols = [f"_{c}_key" for c in keys] + ["_occurrence"]
        ra1 = add_merge_keys(load_ra_sheet(RA1_FILE, spec["ra1_sheet"], spec["ra1_code"]), keys)
        ra2 = add_merge_keys(load_ra_sheet(RA2_FILE, spec["ra2_sheet"], spec["ra2_code"]), keys)
        llm = add_merge_keys(llm_rows, keys)

        merged = ra1.merge(
            ra2[merge_cols + ["ra_code"]].rename(columns={"ra_code": "ra2_code"}),
            on=merge_cols, how="left", validate="one_to_one",
        ).rename(columns={"ra_code": "ra1_code"})
        merged = merged.merge(
            llm[merge_cols + ["source_row", "llm_code"]],
            on=merge_cols, how="left", validate="one_to_one",
        )
        substantive = (merged.ra1_code.isin(CATEGORY_MAP) & merged.ra2_code.isin(CATEGORY_MAP)
                       & merged.llm_code.isin(CATEGORY_MAP))
        llm_matches_ra = merged.llm_code.eq(merged.ra1_code) | merged.llm_code.eq(merged.ra2_code)
        retained = merged.loc[substantive & llm_matches_ra].copy()
        retained["agreement_support"] = "llm_and_one_ra"
        retained.loc[retained.llm_code.eq(retained.ra1_code)
                     & retained.llm_code.eq(retained.ra2_code),
                     "agreement_support"] = "all_three"
        selected.append(retained[["source_row", "ra1_code", "ra2_code", "llm_code",
                                  "agreement_support"]])

    reference = pd.concat(selected, ignore_index=True)
    if reference.source_row.isna().any() or not reference.source_row.is_unique:
        raise ValueError("Reference cases must match unique rows of the LLM workbook.")
    # identifiers and text from the LLM workbook: the RA sheets lack PROLIFIC_PID
    # (market) and game (control)
    reference = reference.merge(
        llm_rows[["source_row", "PROLIFIC_PID", "game", "story", "reasons"]],
        on="source_row", validate="one_to_one",
    )
    reference["training_code"] = reference.llm_code.astype(int)
    reference["training_category"] = reference.training_code.map(CATEGORY_MAP)
    reference = reference.sort_values("source_row").reset_index(drop=True)
    if len(reference) != EXPECTED_REFERENCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_REFERENCE_ROWS} reference examples, "
                         f"found {len(reference)}.")
    OUT.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(reference)} reference examples to {REFERENCE_CSV}")
    print(reference.training_category.value_counts().sort_index().to_string())


# --- shared API plumbing ------------------------------------------------------------
def format_reference_block(reference: pd.DataFrame) -> str:
    pieces = ["Labelled reference examples. Each response has a verified category label."]
    for number, row in enumerate(reference.itertuples(index=False), start=1):
        pieces.append(
            f"Example {number}\n"
            f"Response: <response>{str(row.reasons).strip()}</response>\n"
            f"Category: {row.training_code} ({row.training_category})"
        )
    return "\n\n".join(pieces)


def classify_one(client, model: str, reference_block: str, reason: object,
                 cache: bool) -> str | None:
    query = (
        "Classify this new survey response. Output only the category number.\n"
        f"Response: <response>{str(reason).strip()}</response>"
    )
    block: dict = {"type": "text", "text": reference_block}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    # claude-opus-4-8 rejects the deprecated `temperature` parameter
    kwargs = {} if model.startswith("claude-opus-4-8") else {"temperature": 0.0}
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model, max_tokens=8, system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": [block, {"type": "text", "text": query}]}],
                **kwargs,
            )
            code = re.fullmatch(r"\s*([123])\s*", response.content[0].text)
            if code:
                return code.group(1)
            print(f"Invalid model output on attempt {attempt}: {response.content[0].text!r}")
        except Exception as error:
            print(f"API error on attempt {attempt}: {type(error).__name__}: {error}")
        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
    return None


def get_client():
    import anthropic
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic()


def load_reference() -> pd.DataFrame:
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError("Run --build first.")
    reference = pd.read_csv(REFERENCE_CSV)
    if len(reference) != EXPECTED_REFERENCE_ROWS or not reference.source_row.is_unique:
        raise ValueError("Corrupt reference file.")
    return reference


# --- phase 2: classify the 214 remaining residual answers (AA's script 02) ----------
def classify_remaining(model: str) -> None:
    reference = load_reference()
    llm_rows = load_llm_workbook()
    target = llm_rows.loc[~llm_rows.source_row.isin(reference.source_row)].copy()
    target = target.reset_index(drop=True)
    print(f"{len(target)} targets after excluding {len(reference)} reference rows.")

    out_xlsx = OUT / f"classified_{model}.xlsx"
    code_col, cat_col = "examples_reclassification_code", "examples_reclassified_category"
    target[code_col] = ""
    target[cat_col] = ""
    if out_xlsx.exists():
        prior = pd.read_excel(out_xlsx)[["source_row", code_col, cat_col]]
        prior[code_col] = prior[code_col].astype("string")
        target = target.drop(columns=[code_col, cat_col]).merge(
            prior, on="source_row", how="left", validate="one_to_one")
        target[[code_col, cat_col]] = target[[code_col, cat_col]].fillna("")
        print(f"Resuming from {out_xlsx.name}")

    reference_block = format_reference_block(reference)
    client = get_client()
    for i in target.index:
        if str(target.at[i, code_col]).strip() in VALID_CODES:
            continue
        code = classify_one(client, model, reference_block, target.at[i, "reasons"], cache=True)
        target.at[i, code_col] = code or ""
        target.at[i, cat_col] = CATEGORY_MAP.get(int(code), "") if code else ""
        print(f"source row {target.at[i, 'source_row']} -> {target.at[i, cat_col] or 'FAILED'}")
        if (i + 1) % 25 == 0:
            target.to_excel(out_xlsx, index=False)
    target.to_excel(out_xlsx, index=False)
    n_ok = target[code_col].astype(str).str.strip().isin(VALID_CODES).sum()
    print(f"Finished: {n_ok}/{len(target)} classified -> {out_xlsx}")


# --- phase 3: leave-one-out validation on the 95 ------------------------------------
def loo(model: str) -> None:
    reference = load_reference()
    out_csv = OUT / f"loo_validation_{model}.csv"
    records: list[dict] = []
    if out_csv.exists():
        done = pd.read_csv(out_csv)
        done = done[done.loo_code.astype(str).str.strip().isin(VALID_CODES)]
        records = done.to_dict("records")
    done_rows = {int(r["source_row"]) for r in records}
    client = get_client() if len(done_rows) < len(reference) else None
    for pos in reference.index:
        row = reference.loc[pos]
        if int(row.source_row) in done_rows:
            continue
        block = format_reference_block(reference.drop(index=pos).reset_index(drop=True))
        code = classify_one(client, model, block, row.reasons, cache=False)
        records.append({"source_row": int(row.source_row), "loo_code": code or ""})
        print(f"{len(records)}/95 source row {row.source_row}: "
              f"true {row.training_code} -> pred {code}")
        if len(records) % 10 == 0:
            pd.DataFrame(records).to_csv(out_csv, index=False)
    pd.DataFrame(records).to_csv(out_csv, index=False)

    merged = reference.merge(pd.DataFrame(records), on="source_row", validate="one_to_one")
    merged["loo_code"] = merged.loo_code.astype(int)
    acc = merged.loo_code.eq(merged.training_code).mean()
    a, b = merged.training_code, merged.loo_code
    pe = sum((a == k).mean() * (b == k).mean() for k in CATEGORY_MAP)
    kappa = (acc - pe) / (1 - pe)
    print(f"\nLOO {model}: N={len(merged)}, accuracy {acc:.3f}, kappa {kappa:.3f}")
    print(pd.crosstab(merged.training_category, merged.loo_code).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--loo", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args()
    if args.build:
        build_reference()
    if args.classify:
        classify_remaining(args.model)
    if args.loo:
        loo(args.model)
    if not (args.build or args.classify or args.loo):
        parser.error("Pass at least one of --build / --classify / --loo.")


if __name__ == "__main__":
    main()
