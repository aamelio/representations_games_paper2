import csv
from pathlib import Path


WORK = Path(__file__).resolve().parent
SOURCE = WORK / "llm_batch_3.csv"
LABELS = WORK / "batch3_part_3134_4266_labels.txt"
OUTPUT = WORK / "llm_batch_3_part_3134_4266.csv"
START = 3134
STOP = 4267
ORIGIN = "gpt_5_6_subagent_rowwise"


with SOURCE.open(newline="", encoding="utf-8-sig") as source_file:
    source_rows = list(csv.DictReader(source_file))

labels = "".join(LABELS.read_text(encoding="utf-8").split())
selected_rows = source_rows[START:STOP]

if len(selected_rows) != STOP - START:
    raise ValueError(f"Expected {STOP - START} source rows, found {len(selected_rows)}")
if len(labels) != len(selected_rows):
    raise ValueError(f"Expected {len(selected_rows)} labels, found {len(labels)}")
if not set(labels) <= set("0123"):
    raise ValueError("Labels must be integers 0-3")

with OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=["classification_id", "category_num", "classification_origin"],
    )
    writer.writeheader()
    writer.writerows(
        {
            "classification_id": row["classification_id"],
            "category_num": label,
            "classification_origin": ORIGIN,
        }
        for row, label in zip(selected_rows, labels)
    )
