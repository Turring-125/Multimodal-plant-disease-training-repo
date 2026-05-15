"""
Fix dataset_index.json so every crop×disease class has samples in train/val/test.

The original split was done per-district, which caused all 7 cotton classes to land
entirely in train (cotton uses only 3 Indian states whose districts all fell in train).

Strategy: Stratified split per class, preserving 70/15/15 ratio.
"""
import json
import random
from collections import defaultdict

random.seed(42)

with open("dataset/dataset_index.json") as f:
    records = json.load(f)

# Group records by class
class_records = defaultdict(list)
for r in records:
    key = f"{r['crop_type']}_{r['disease_label']}"
    class_records[key].append(r)

print("Before fix:")
for cls in sorted(class_records):
    splits = defaultdict(int)
    for r in class_records[cls]:
        splits[r["split"]] += 1
    print(f"  {cls:40s}  train={splits.get('train',0):4d}  val={splits.get('val',0):4d}  test={splits.get('test',0):4d}")

# Re-assign splits per class
for cls, recs in class_records.items():
    random.shuffle(recs)
    n = len(recs)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)
    
    for i, r in enumerate(recs):
        if i < train_end:
            r["split"] = "train"
        elif i < val_end:
            r["split"] = "val"
        else:
            r["split"] = "test"

# Flatten back
all_records = []
for recs in class_records.values():
    all_records.extend(recs)

print(f"\nAfter fix ({len(all_records)} total):")
# Re-group to verify
class_records2 = defaultdict(list)
for r in all_records:
    key = f"{r['crop_type']}_{r['disease_label']}"
    class_records2[key].append(r)

for cls in sorted(class_records2):
    splits = defaultdict(int)
    for r in class_records2[cls]:
        splits[r["split"]] += 1
    print(f"  {cls:40s}  train={splits.get('train',0):4d}  val={splits.get('val',0):4d}  test={splits.get('test',0):4d}")

with open("dataset/dataset_index.json", "w") as f:
    json.dump(all_records, f, indent=2)

print("\nSaved fixed dataset_index.json")
