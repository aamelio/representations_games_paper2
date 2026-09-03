import csv
from pathlib import Path


work_dir = Path(__file__).resolve().parent
source_path = work_dir / "llm_batch_2.csv"
labels_path = work_dir / "batch2_labels_checkpoint.txt"
output_path = work_dir / "llm_batch_2_classified.csv"

labels = "".join(labels_path.read_text(encoding="utf-8").split())

with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
    source_rows = list(csv.DictReader(source_file))

if len(labels) != len(source_rows):
    raise ValueError(f"Label/source mismatch: {len(labels)} labels, {len(source_rows)} rows")
if any(label not in "0123" for label in labels):
    raise ValueError("Checkpoint contains a label outside 0-3")

with output_path.open("w", encoding="utf-8", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=["classification_id", "category_num", "classification_origin"],
    )
    writer.writeheader()
    for source_row, label in zip(source_rows, labels):
        writer.writerow(
            {
                "classification_id": source_row["classification_id"],
                "category_num": label,
                "classification_origin": "gpt_5_6_subagent_rowwise",
            }
        )
