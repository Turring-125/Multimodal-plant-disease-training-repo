"""
check_data.py — v3

Verifies the dataset_index.json and weather files are consistent:
  - All weather .npy files referenced in the index exist
  - All .npy files have the correct shape [7, 11]
  - No duplicate image paths
  - Split distribution check
  - Per-crop / per-disease count summary
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import numpy as np
from collections import Counter, defaultdict

PROJECTV3_PATH = r"d:\clg\major project\Project folder\projectv3"
DATASET_DIR    = os.path.join(PROJECTV3_PATH, "dataset")
INDEX_PATH     = os.path.join(DATASET_DIR, "dataset_index.json")
WEATHER_DIR    = os.path.join(DATASET_DIR, "weather")

EXPECTED_SHAPE = (7, 11)


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] Index not found: {INDEX_PATH}")
        return

    with open(INDEX_PATH) as f:
        records = json.load(f)

    print(f"Total records: {len(records)}")

    # ── Duplicate image paths ─────────────────────────────────────────────
    paths = [r["image_path"] for r in records]
    dup_count = len(paths) - len(set(paths))
    if dup_count:
        print(f"[WARN] {dup_count} duplicate image paths found!")
    else:
        print("✓ No duplicate image paths")

    # ── Missing / bad-shape weather files ─────────────────────────────────
    missing     = []
    wrong_shape = []
    for r in records:
        wf   = r.get("weather_file", f"{r['id']}_weather.npy")
        path = os.path.join(WEATHER_DIR, wf)
        if not os.path.exists(path):
            missing.append(r["id"])
        else:
            arr = np.load(path)
            if arr.shape != EXPECTED_SHAPE:
                wrong_shape.append((r["id"], arr.shape))

    if missing:
        print(f"[ERROR] {len(missing)} missing weather files: {missing[:10]}...")
    else:
        print(f"✓ All {len(records)} weather files present")

    if wrong_shape:
        print(f"[ERROR] {len(wrong_shape)} wrong-shape files:")
        for sid, shape in wrong_shape[:10]:
            print(f"    {sid}: {shape} (expected {EXPECTED_SHAPE})")
    else:
        print(f"✓ All weather files have correct shape {EXPECTED_SHAPE}")

    # ── Split distribution ────────────────────────────────────────────────
    splits = Counter(r["split"] for r in records)
    print("\n── Split distribution ──")
    for split in ("train", "val", "test"):
        n = splits.get(split, 0)
        pct = 100 * n / max(len(records), 1)
        print(f"  {split:6s}: {n:6d} ({pct:5.1f}%)")

    # ── Crop distribution ─────────────────────────────────────────────────
    crops = Counter(r["crop_type"] for r in records)
    print("\n── Crop distribution ──")
    for crop, n in sorted(crops.items()):
        print(f"  {crop:12s}: {n}")

    # ── Per-crop disease counts ───────────────────────────────────────────
    print("\n── Disease counts per crop ──")
    disease_by_crop = defaultdict(Counter)
    for r in records:
        disease_by_crop[r["crop_type"]][r["disease_label"]] += 1
    for crop in sorted(disease_by_crop):
        print(f"\n  {crop.upper()}")
        for disease, count in sorted(disease_by_crop[crop].items(),
                                      key=lambda x: -x[1]):
            print(f"    {disease:35s}: {count}")

    # ── Weather stats file ────────────────────────────────────────────────
    stats_path = os.path.join(DATASET_DIR, "weather_stats.npy")
    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()
        print(f"\n✓ weather_stats.npy present (mean shape: {stats['mean'].shape})")
    else:
        print("\n[WARN] weather_stats.npy NOT found — run compute_stats.py!")


if __name__ == "__main__":
    main()
