import os
import csv

dataset_root = "leaks_dataset"
csv_filename = "clean_metadata.csv"

rows = [("filepath", "leak_type", "source")]  # source = real | synthetic

for leak_type in ["water_leak", "oil_leak"]:
    for source in ["real", "synthetic"]:
        folder = os.path.join(dataset_root, leak_type, source)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(folder, fname)
                rows.append((filepath, leak_type, source))

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f" Metadata saved → {csv_filename}")
print(f" Total images indexed: {len(rows) - 1}")

