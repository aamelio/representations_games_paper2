#!/usr/bin/env python3
"""Round-3 pilot: aggregate the draft-pool ratings into discussion tables.

PRE-FREEZE PILOT OUTPUT --- for the AA/NG design discussion only; not a paper exhibit.

Sender side: vignette ids encode category (M/S/C) and belief level (H/L). Per context,
the category similarity is the mean over the category's 16 vignettes (both belief
levels) x 3 conversations; the belief margin is mean(High 8) - mean(Low 8), the new
similarity measure for the belief component of the representation. Receiver side: the
four classes (G/B/S/C) x the four strategic contexts, graded means over 4 vignettes x 3
conversations.

Inputs:  pilot_recording.csv, pilot_retrieval.csv, pilot_receiver.csv
Output:  pilot_tables.txt
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

CAT = {"M": "Moral", "S": "Self-interest", "C": "Mutual Benefit/Coop."}
RCLASS = {"G": "Moral good", "B": "Moral bad", "S": "Self-interest", "C": "Mutual Benefit"}
CTX_ORDER = ["C-KW", "M-KW", "C-LT", "M-LT", "C-UG", "M-UG", "C-TG", "M-TG",
             "BONUS", "AID"]
GAMES = ["KW", "LT", "UG", "TG"]

LOG: list[str] = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def main() -> None:
    rec = pd.read_csv(HERE / "pilot_recording.csv")
    ret = pd.read_csv(HERE / "pilot_retrieval.csv")
    rcv = pd.read_csv(HERE / "pilot_receiver.csv")
    rec["category"] = rec["vignette_id"].str[0].map(CAT)
    rec["belief"] = rec["vignette_id"].str[1]
    ret["category"] = ret["vignette_id"].str[0].map(CAT)

    log("ROUND-3 PILOT (draft pool, PRE-FREEZE) --- rater Opus 4.8, 3 permuted conversations\n")

    log("=== Sender: category similarity (mean over 16 vignettes x 3 conversations) ===")
    cat = (rec.groupby(["context", "category"])["rating"].mean()
              .unstack().reindex(CTX_ORDER).round(1))
    log(cat.to_string())

    log("\n=== Sender: belief margin (High minus Low, by category) ===")
    bel = (rec.groupby(["context", "category", "belief"])["rating"].mean()
              .unstack("belief"))
    bel = (bel["H"] - bel["L"]).unstack().reindex(CTX_ORDER).round(1)
    log(bel.to_string())

    log("\n=== Market - Control deltas, per game ===")
    for name, table in [("category similarity", cat), ("belief margin", bel)]:
        log(f"  {name}:")
        for g in GAMES:
            d = (table.loc[f"M-{g}"] - table.loc[f"C-{g}"]).round(1)
            log(f"    {g:3} " + "  ".join(f"{c.split()[0]} {d[c]:+.1f}" for c in table.columns))

    log("\n=== Stories: Aid minus Bonus ===")
    for name, table in [("category similarity", cat), ("belief margin", bel)]:
        d = (table.loc["AID"] - table.loc["BONUS"]).round(1)
        log(f"  {name}: " + "  ".join(f"{c.split()[0]} {d[c]:+.1f}" for c in table.columns))

    log("\n=== Retrieval split: category share in percent (mean over conversations) ===")
    tot = ret.groupby(["set", "context"])["points"].transform("sum")
    ret["share"] = ret["points"] / tot * 100
    sh = ret.groupby(["context", "category"])["share"].sum().unstack() / 3
    log(sh.reindex(CTX_ORDER).round(1).to_string())

    log("\n=== Receiver: class similarity, strategic contexts (graded; 4 vignettes x 3 conv.) ===")
    g = rcv[rcv.task == "graded"].copy()
    g["rclass"] = g["receiver_id"].str[0].map(RCLASS)
    rg = (g.groupby(["context", "rclass"])["value"].mean()
            .unstack().reindex(["C-UG", "M-UG", "C-TG", "M-TG"]).round(1))
    log(rg.to_string())
    log("\n  Market - Control:")
    for game in ["UG", "TG"]:
        d = (rg.loc[f"M-{game}"] - rg.loc[f"C-{game}"]).round(1)
        log(f"    {game}: " + "  ".join(f"{c} {d[c]:+.1f}" for c in rg.columns))

    log("\n=== Receiver: class share of the split in percent ===")
    s = rcv[rcv.task == "split"].copy()
    s["rclass"] = s["receiver_id"].str[0].map(RCLASS)
    stot = s.groupby(["set", "context"])["value"].transform("sum")
    s["share"] = s["value"] / stot * 100
    rs = s.groupby(["context", "rclass"])["share"].sum().unstack() / 3
    log(rs.reindex(["C-UG", "M-UG", "C-TG", "M-TG"]).round(1).to_string())

    (HERE / "pilot_tables.txt").write_text("\n".join(LOG))
    print(f"\nwrote {HERE / 'pilot_tables.txt'}")


if __name__ == "__main__":
    main()
