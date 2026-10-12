"""
compute_stats.py — v3

Computes z-score normalization statistics (mean, std) for the 11 weather
features using ONLY the training split records.

Run this AFTER data_collection.py completes and BEFORE training.

Output: dataset/weather_stats.npy  (shape: dict with keys 'mean', 'std', each [11])
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import numpy as np
from tqdm import tqdm

PROJECTV3_PATH = r"d:\clg\major project\Project folder\projectv3"
DATASET_DIR    = os.path.join(PROJECTV3_PATH, "dataset")
INDEX_PATH     = os.path.join(DATASET_DIR, "dataset_index.json")
WEATHER_DIR    = os.path.join(DATASET_DIR, "weather")
STATS_PATH     = os.path.join(DATASET_DIR, "weather_stats.npy")

WINDOW_DAYS = 7
N_FEATURES  = 11


def main():
    print(f"Loading index from: {INDEX_PATH}")
    with open(INDEX_PATH) as f:
        records = json.load(f)

    train_records = [r for r in records if r["split"] == "train"]
    print(f"Train records: {len(train_records)} / {len(records)} total")

    if not train_records:
        raise RuntimeError("No training records found! Run data_collection.py first.")

    # Collect all training weather arrays
    all_arrays = []
    missing = 0
    for r in tqdm(train_records, desc="Loading weather files"):
        wf = r.get("weather_file", f"{r['id']}_weather.npy")
        path = os.path.join(WEATHER_DIR, wf)
        if not os.path.exists(path):
            missing += 1
            continue
        arr = np.load(path).astype(np.float32)  # [7, 11]
        if arr.shape != (WINDOW_DAYS, N_FEATURES):
            print(f"  [WARN] Unexpected shape {arr.shape} for {path}. Skipping.")
            continue
        all_arrays.append(arr)

    if missing > 0:
        print(f"[WARN] {missing} weather files not found — run data_collection.py to fix.")

    if not all_arrays:
        raise RuntimeError("No valid weather files found for training split.")

    stack = np.stack(all_arrays, axis=0)  # [N_train, 7, 11]
    mean  = stack.mean(axis=(0, 1))       # [11]
    std   = stack.std(axis=(0, 1))        # [11]
    std   = np.where(std < 1e-8, 1e-8, std)  # avoid division by zero

    np.save(STATS_PATH, {"mean": mean, "std": std})
    print(f"\nStats saved to: {STATS_PATH}")

    feature_names = [
        "temp_max", "temp_min", "temp_mean", "rh_max", "rh_mean",
        "precip_sum", "dewpoint_mean", "wind_max", "soil_temp",
        "month_sin", "month_cos"
    ]
    print("\n── Per-feature statistics (train) ──")
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 42)
    for name, m, s in zip(feature_names, mean, std):
        print(f"{name:<20} {m:>10.4f} {s:>10.4f}")


if __name__ == "__main__":
    main()
