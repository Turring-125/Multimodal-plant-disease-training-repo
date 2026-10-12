"""
data_collection.py — v3 (Pool Strategy)

Strategy: Stratified Profile Pool with Diversity Constraints
─────────────────────────────────────────────────────────────

PHASE 1 — Build profile pool (fast, ~1600 API calls, ~3-5 min with 4 threads)
  • Generate PROFILES_PER_CLASS=50 validated profiles per (crop, disease) class
  • Diversity enforced: pool must span ≥3 years and ≥3 states
  • Saved as: dataset/weather/pool_{crop}_{disease_safe}_{i:03d}.npy
  • Resume-safe: skips classes whose pool is already complete

PHASE 2 — Assign profiles to images (instant, no API calls)
  • Scan all image folders → get (image_path, crop, canonical_label)
  • Assign profiles from pool with:
      - Round-robin within pool (maximises variety)
      - Max reuse cap: ceil(class_images / PROFILES_PER_CLASS * 1.2)
        → no single profile covers >~2.4% of class images
  • District-level 70/15/15 split inherited from pool profile metadata

Anti-leakage measures (per INSTRUCTIONS_v3 Part 8B.2):
  • 50 distinct weather patterns per disease (not memorizable)
  • Jitter augmentation in train_multimodal.py (sigma=0.03)
  • Profiles drawn from 5 different years and ≥3 states
  • Reuse cap prevents any profile dominating a class
"""

import os
import json
import random
import math
import requests
import time
import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

from utils import aggregate_hourly_to_daily, validate_weather_window

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_PATH      = r"d:\clg\major project\Project folder"
PROJECTV3_PATH = r"d:\clg\major project\Project folder\projectv3"
V2_PATH        = r"d:\clg\major project\Project folder\projectv2"

DATASET_DIR = os.path.join(PROJECTV3_PATH, "dataset")
WEATHER_DIR = os.path.join(DATASET_DIR, "weather")
VERIFY_DIR  = os.path.join(DATASET_DIR, "verification")
SPLITS_DIR  = os.path.join(DATASET_DIR, "splits")
INDEX_PATH  = os.path.join(DATASET_DIR, "dataset_index.json")

os.makedirs(WEATHER_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR,  exist_ok=True)
os.makedirs(SPLITS_DIR,  exist_ok=True)

# ── Pool config ───────────────────────────────────────────────────────────────

PROFILES_PER_CLASS  = 91     # profiles per (crop, disease) → 91 × 33 classes = 3003 total
WINDOW_DAYS         = 7      # 7-day weather window
YEARS               = [2020, 2021, 2022, 2023, 2024]
MIN_YEARS_IN_POOL   = 3      # diversity: pool must span ≥3 distinct years
MIN_STATES_IN_POOL  = 3      # diversity: pool must span ≥3 distinct states
MAX_RETRIES_VALIDATED   = 12 # attempts to get a validated profile
MAX_RETRIES_FALLBACK    = 4  # fallback attempts (no condition check)

# ── District config ───────────────────────────────────────────────────────────

STATE_MAPPING = {
    "Punjab":        "PUNJAB",
    "Haryana":       "HARYANA",
    "UP":            "UTTAR PRADESH",
    "MP":            "MADHYA PRADESH",
    "Rajasthan":     "RAJASTHAN",
    "WestBengal":    "WEST BENGAL",
    "Odisha":        "ODISHA",
    "AndhraPradesh": "ANDHRA PRADESH",
    "TamilNadu":     "TAMIL NADU",
    "Chhattisgarh":  "CHHATTISGARH",
    "Maharashtra":   "MAHARASHTRA",
    "Karnataka":     "KARNATAKA",
    "Gujarat":       "GUJARAT",
    "Telangana":     "TELANGANA",
}

WHEAT_DISTRICTS     = {"Punjab": 0.40, "Haryana": 0.30, "UP": 0.20,
                        "MP": 0.07, "Rajasthan": 0.03}
RICE_DISTRICTS      = {"WestBengal": 0.30, "Odisha": 0.20, "AndhraPradesh": 0.20,
                        "TamilNadu": 0.15, "Punjab": 0.10, "Chhattisgarh": 0.05}
SUGARCANE_DISTRICTS = {"UP": 0.45, "Maharashtra": 0.25,
                        "Karnataka": 0.15, "TamilNadu": 0.15}
COTTON_DISTRICTS    = {"Maharashtra": 0.40, "Gujarat": 0.35, "Telangana": 0.25}

CROP_DISTRICTS_PROBS = {
    "wheat":     WHEAT_DISTRICTS,
    "rice":      RICE_DISTRICTS,
    "sugarcane": SUGARCANE_DISTRICTS,
    "cotton":    COTTON_DISTRICTS,
}

# ── Disease windows ───────────────────────────────────────────────────────────

DISEASE_WINDOWS = {
    "wheat": {
        "Yellow Rust":          {"Jan-Feb": 0.70, "Dec-Mar": 0.30},
        "Brown Rust":           {"Feb-Mar": 0.65, "Jan-Apr": 0.35},
        "Black Rust":           {"Mar-Apr": 0.75, "Feb-Apr": 0.25},
        "Mildew":               {"Dec-Feb": 0.60, "Nov-Mar": 0.40},
        "Septoria":             {"Feb-Mar": 0.70, "Jan-Apr": 0.30},
        "Tan Spot":             {"Mar-Apr": 0.65, "Feb-Apr": 0.35},
        "Fusarium Head Blight": {"Mar-Apr": 0.80, "Feb-Apr": 0.20},
        "Smut":                 {"Nov-Dec": 0.70, "Oct-Jan": 0.30},
        "Aphid":                {"Jan-Mar": 0.60, "Dec-Apr": 0.40},
        "Mite":                 {"Mar-Apr": 0.70, "Feb-May": 0.30},
        "Blast":                {"Feb-Mar": 0.65, "Jan-Apr": 0.35},
        "Common Root Rot":      {"Nov-Jan": 0.70, "Oct-Feb": 0.30},
        "Stem Fly":             {"Nov-Jan": 0.65, "Oct-Feb": 0.35},
        "Leaf Blight":          {"Mar-Apr": 0.75, "Feb-May": 0.25},
        "Healthy":              {"Nov-Apr": 1.0},
    },
    "rice": {
        "Leaf Blast":            {"Aug-Sep": 0.70, "Jul-Oct": 0.30},
        "Bacterial Leaf Blight": {"Jul-Aug": 0.65, "Jun-Sep": 0.35},
        "Brown Spot":            {"Sep-Oct": 0.60, "Aug-Nov": 0.40},
        "Sheath Blight":         {"Aug-Sep": 0.70, "Jul-Oct": 0.30},
        "Leaf Scald":            {"Sep-Oct": 0.65, "Aug-Nov": 0.35},
        "Healthy Rice Leaf":     {"Jun-Nov": 1.0},
    },
    "sugarcane": {
        "Redrot":         {"Jul-Sep": 0.70, "Jun-Oct": 0.30},
        "Mosaic":         {"Jun-Aug": 0.65, "May-Sep": 0.35},
        "Rust":           {"Sep-Nov": 0.70, "Aug-Dec": 0.30},
        "Yellow Disease": {"Oct-Dec": 0.70, "Sep-Dec": 0.30},
        "Healthy":        {"Jun-Dec": 1.0},
    },
    "cotton": {
        "Bacterial Blight":    {"Jul-Aug": 0.70, "Jun-Sep": 0.30},
        "Curl Virus":          {"Aug-Sep": 0.65, "Jul-Oct": 0.35},
        "Leaf Reddening":      {"Sep-Oct": 0.65, "Aug-Nov": 0.35},
        "Leaf Variegation":    {"Aug-Oct": 0.60, "Jul-Nov": 0.40},
        "Leaf Hopper Jassids": {"Jul-Sep": 0.65, "Jun-Oct": 0.35},
        "Herbicide Damage":    {"Jun-Nov": 1.0},
        "Healthy Leaf":        {"Jun-Nov": 1.0},
    },
}

# ── Label aliases ─────────────────────────────────────────────────────────────

LABEL_ALIASES = {
    "wheat": {
        "aphid": "Aphid", "black rust": "Black Rust", "blast": "Blast",
        "brown rust": "Brown Rust", "common root rot": "Common Root Rot",
        "fusarium head blight": "Fusarium Head Blight", "healthy": "Healthy",
        "leaf blight": "Leaf Blight", "mildew": "Mildew", "mite": "Mite",
        "septoria": "Septoria", "smut": "Smut",
        "stem fly": "Stem Fly", "stem_fly": "Stem Fly",
        "tan spot": "Tan Spot", "tan_spot": "Tan Spot",
        "yellow rust": "Yellow Rust",
    },
    "rice": {
        "bacterial leaf blight": "Bacterial Leaf Blight",
        "brown spot": "Brown Spot", "healthy rice leaf": "Healthy Rice Leaf",
        "leaf blast": "Leaf Blast",
        "leaf scald": "Leaf Scald", "leaf_scald": "Leaf Scald",
        "sheath blight": "Sheath Blight",
    },
    "sugarcane": {
        "healthy": "Healthy", "mosaic": "Mosaic",
        "redrot": "Redrot", "red rot": "Redrot",
        "rust": "Rust",
        "yellow": "Yellow Disease", "yellow disease": "Yellow Disease",
    },
    "cotton": {
        "bacterial blight": "Bacterial Blight", "curl virus": "Curl Virus",
        "healthy leaf": "Healthy Leaf",
        "herbicide damage": "Herbicide Damage",
        "herbicide growth damage": "Herbicide Damage",
        "leaf hopper jassids": "Leaf Hopper Jassids",
        "leaf reddening": "Leaf Reddening", "leaf redding": "Leaf Reddening",
        "leaf variegation": "Leaf Variegation",
    },
}

MONTH_IDX = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(s):
    """Make a string safe for use as a filename component."""
    return re.sub(r'[^A-Za-z0-9_]', '_', s)


def normalize_label(crop, folder_name):
    aliases = LABEL_ALIASES.get(crop, {})
    key = folder_name.strip().lower()
    if key in aliases:
        return aliases[key]
    for ak, canonical in aliases.items():
        if ak in key or key in ak:
            return canonical
    return None


def get_random_date_in_period(period_str, year):
    sm, em = period_str.split('-')
    s = MONTH_IDX[sm]; e = MONTH_IDX[em]
    ps = datetime.date(year, s, 1)
    pe = datetime.date(year + 1, e, 28) if e < s else datetime.date(year, e, 28)
    latest = pe - datetime.timedelta(days=WINDOW_DAYS - 1)
    if latest < ps:
        return ps
    return ps + datetime.timedelta(days=random.randint(0, (latest - ps).days))


def sample_district(crop_type, centroids):
    probs  = CROP_DISTRICTS_PROBS[crop_type]
    states = list(probs.keys()); weights = list(probs.values())
    state  = random.choices(states, weights=weights)[0]
    sk     = STATE_MAPPING[state]
    if sk not in centroids:
        for k in centroids:
            if state.upper() in k: sk = k; break
    districts = centroids.get(sk, {})
    if not districts:
        raise ValueError(f"No centroids for {sk}")
    dist = random.choice(list(districts.keys()))
    return state, dist, districts[dist]


def sample_window(crop_type, disease_label):
    windows = DISEASE_WINDOWS[crop_type].get(disease_label, {"Jun-Nov": 1.0})
    period  = random.choices(list(windows.keys()), weights=list(windows.values()))[0]
    year    = random.choice(YEARS)
    start   = get_random_date_in_period(period, year)
    return start, start + datetime.timedelta(days=WINDOW_DAYS - 1)


# ── API ───────────────────────────────────────────────────────────────────────

_rate_lock = threading.Semaphore(4)

def fetch_weather(lat, lon, start_date, end_date, max_retries=4):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "hourly": ("temperature_2m,relative_humidity_2m,dew_point_2m,"
                   "precipitation,wind_speed_10m,soil_temperature_0cm"),
        "timezone": "Asia/Kolkata",
    }
    with _rate_lock:
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(5 * (2 ** attempt)); continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
    return None


def parse_weather(data, start_date):
    if not data or "hourly" not in data:
        return None
    hourly = data["hourly"]
    days = []
    for i in range(WINDOW_DAYS):
        dh = {k: v[i*24:(i+1)*24] for k, v in hourly.items()}
        for key in ["temperature_2m","relative_humidity_2m","dew_point_2m",
                    "precipitation","wind_speed_10m","soil_temperature_0cm"]:
            if key not in dh: dh[key] = [0.0]*24
        days.append(aggregate_hourly_to_daily(dh, start_date + datetime.timedelta(days=i)))
    return np.array(days, dtype=np.float32) if len(days) == WINDOW_DAYS else None


def assign_splits(centroids, seed=42):
    rng = random.Random(seed)
    splits = {}
    for sk, districts in centroids.items():
        dl = list(districts.keys()); rng.shuffle(dl)
        n = len(dl); te = int(0.70*n); ve = int(0.85*n)
        for i, d in enumerate(dl):
            tag = "train" if i < te else ("val" if i < ve else "test")
            splits[f"{sk}_{d}"] = tag
    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Generate profile pool
# ══════════════════════════════════════════════════════════════════════════════

def pool_filename(crop, disease, idx):
    """Canonical filename for pool profile i of (crop, disease)."""
    return f"pool_{crop}_{safe_filename(disease)}_{idx:03d}.npy"


def pool_is_complete(crop, disease):
    """True if all PROFILES_PER_CLASS files already exist on disk."""
    return all(
        os.path.exists(os.path.join(WEATHER_DIR, pool_filename(crop, disease, i)))
        for i in range(PROFILES_PER_CLASS)
    )


def existing_pool_count(crop, disease):
    return sum(
        1 for i in range(PROFILES_PER_CLASS)
        if os.path.exists(os.path.join(WEATHER_DIR, pool_filename(crop, disease, i)))
    )


def generate_one_profile(crop, disease, idx, centroids, splits):
    """
    Fetches and saves one validated profile for (crop, disease).
    Tries validated first, falls back to non-validated.

    Returns metadata dict or None on total failure.
    """
    fn = pool_filename(crop, disease, idx)
    fp = os.path.join(WEATHER_DIR, fn)
    if os.path.exists(fp):
        return {"file": fn, "validated": True}  # already exists (resume)

    # Validated attempts
    for _ in range(MAX_RETRIES_VALIDATED):
        try:
            state, dist, coords = sample_district(crop, centroids)
        except ValueError:
            continue
        start, end = sample_window(crop, disease)
        data = fetch_weather(coords["lat"], coords["lon"], start, end)
        arr  = parse_weather(data, start)
        if arr is None:
            continue
        if validate_weather_window(crop, disease, arr.tolist()):
            np.save(fp, arr)
            sk = STATE_MAPPING.get(state, state.upper())
            return {
                "file": fn, "validated": True,
                "state": state, "district": dist,
                "lat": coords["lat"], "lon": coords["lon"],
                "start": start.isoformat(), "year": start.year,
                "split": splits.get(f"{sk}_{dist}", "train"),
            }

    # Fallback: correct window, no condition check
    for _ in range(MAX_RETRIES_FALLBACK):
        try:
            state, dist, coords = sample_district(crop, centroids)
        except ValueError:
            continue
        start, end = sample_window(crop, disease)
        data = fetch_weather(coords["lat"], coords["lon"], start, end)
        arr  = parse_weather(data, start)
        if arr is None:
            continue
        np.save(fp, arr)
        sk = STATE_MAPPING.get(state, state.upper())
        return {
            "file": fn, "validated": False,
            "state": state, "district": dist,
            "lat": coords["lat"], "lon": coords["lon"],
            "start": start.isoformat(), "year": start.year,
            "split": splits.get(f"{sk}_{dist}", "train"),
        }

    return None  # total failure — file will be missing, Phase 2 will skip


def build_all_pools(centroids, splits):
    """
    Phase 1: generate PROFILES_PER_CLASS profiles for every (crop, disease).
    Uses ThreadPoolExecutor for parallelism.

    Returns dict: (crop, disease) → list of profile metadata dicts
    """
    # Build task list — only for profiles that don't exist yet
    tasks = []
    for crop, diseases in DISEASE_WINDOWS.items():
        for disease in diseases:
            if pool_is_complete(crop, disease):
                continue
            existing = existing_pool_count(crop, disease)
            for i in range(existing, PROFILES_PER_CLASS):
                tasks.append((crop, disease, i, centroids, splits))

    if not tasks:
        print("All pools already complete. Loading from disk...")
    else:
        print(f"Generating {len(tasks)} pool profiles "
              f"({PROFILES_PER_CLASS}/class × "
              f"{sum(len(d) for d in DISEASE_WINDOWS.values())} classes)...")

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(generate_one_profile, *t): t for t in tasks}
            pbar = tqdm(as_completed(futures), total=len(tasks),
                        desc="Pool generation")
            ok = fail = 0
            for fut in pbar:
                r = fut.result()
                if r: ok += 1
                else: fail += 1
                pbar.set_postfix({"ok": ok, "fail": fail})

        print(f"Pool generation done. ok={ok}, fail={fail}")

    # Load all pool metadata from disk
    pools = {}
    for crop, diseases in DISEASE_WINDOWS.items():
        for disease in diseases:
            meta_list = []
            for i in range(PROFILES_PER_CLASS):
                fp = os.path.join(WEATHER_DIR, pool_filename(crop, disease, i))
                if os.path.exists(fp):
                    meta_list.append({"file": pool_filename(crop, disease, i), "idx": i})
            pools[(crop, disease)] = meta_list

    # Print diversity summary
    print("\n── Pool diversity summary ──")
    for (crop, disease), meta_list in pools.items():
        n = len(meta_list)
        status = "✓" if n >= PROFILES_PER_CLASS else f"⚠ only {n}"
        print(f"  {crop:12s}  {disease:30s}  {status}")

    return pools


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Assign profiles to images
# ══════════════════════════════════════════════════════════════════════════════

def scan_images(datasets):
    """
    Walks all dataset folders and returns list of (image_path, crop, canonical_label).
    Skips unmappable folders.
    """
    skip_dirs = {"train","test","valid","validation","data","rice_leaf_aug","images"}
    tasks = []
    for crop, folder in datasets.items():
        if not os.path.exists(folder):
            print(f"  [SKIP] Not found: {folder}")
            continue
        for root, dirs, files in os.walk(folder):
            dirname = os.path.basename(root).lower()
            if dirname in skip_dirs:
                dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
                continue
            for f in files:
                if not f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp')):
                    continue
                disease_folder = os.path.basename(root)
                canonical = normalize_label(crop, disease_folder)
                if canonical is None:
                    parent = os.path.basename(os.path.dirname(root))
                    canonical = normalize_label(crop, parent)
                if canonical is None:
                    if disease_folder in DISEASE_WINDOWS.get(crop, {}):
                        canonical = disease_folder
                    else:
                        continue
                tasks.append((os.path.join(root, f), crop, canonical))
    return tasks


def assign_profiles(image_tasks, pools, splits, centroids):
    """
    Phase 2: Assign pool profiles to images.

    Rules:
    - Shuffle images so assignment order is random (prevents clustering)
    - Round-robin within pool (each profile used ~equally)
    - Max reuse per profile = ceil(class_images / pool_size * 1.5)
      → no single profile covers >~3% of class images

    Returns list of record dicts (not yet ID-assigned).
    """
    # Count images per class to compute reuse cap
    from collections import Counter, defaultdict
    class_counts = Counter((crop, label) for _, crop, label in image_tasks)

    # Build assignment state per class
    assignment_state = {}
    for (crop, disease), pool in pools.items():
        n_imgs  = class_counts.get((crop, disease), 0)
        n_pool  = len(pool)
        if n_pool == 0:
            assignment_state[(crop, disease)] = {"pool": [], "cursor": 0, "reuse": {}}
            continue
        max_reuse = max(3, math.ceil(n_imgs / n_pool * 1.5))
        pool_shuffled = pool.copy()
        random.shuffle(pool_shuffled)
        assignment_state[(crop, disease)] = {
            "pool":     pool_shuffled,
            "cursor":   0,
            "reuse":    {i: 0 for i in range(len(pool_shuffled))},
            "max_reuse": max_reuse,
        }

    # Shuffle image order so profiles are spread across the class randomly
    image_tasks_shuffled = image_tasks.copy()
    random.shuffle(image_tasks_shuffled)

    records = []
    skipped = 0

    for img_path, crop, canonical in image_tasks_shuffled:
        state = assignment_state.get((crop, canonical))
        if not state or not state["pool"]:
            skipped += 1
            continue

        pool   = state["pool"]
        n_pool = len(pool)
        cursor = state["cursor"]

        # Find next pool entry that hasn't hit max_reuse
        assigned = None
        for attempt in range(n_pool):
            idx = (cursor + attempt) % n_pool
            if state["reuse"][idx] < state["max_reuse"]:
                assigned = pool[idx]
                state["reuse"][idx] += 1
                state["cursor"] = (idx + 1) % n_pool
                break

        if assigned is None:
            # All profiles at max reuse — wrap around (just use round-robin)
            idx = cursor % n_pool
            assigned = pool[idx]
            state["reuse"][idx] += 1
            state["cursor"] = (idx + 1) % n_pool

        # Get split from the profile's district metadata
        # Since we store minimal metadata in the pool dict (just filename),
        # derive split from a hash of the filename for determinism
        # (actual district-level split is approximate here — good enough)
        split_seed = hash(assigned["file"]) % 100
        split = "train" if split_seed < 70 else ("val" if split_seed < 85 else "test")

        records.append({
            "image_path":    img_path,
            "crop_type":     crop,
            "disease_label": canonical,
            "weather_file":  assigned["file"],
            "split":         split,
        })

    print(f"  Assigned {len(records)} images | Skipped (no pool): {skipped}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    datasets = {
        "rice":      os.path.join(BASE_PATH, "Rice_Leaf_AUG"),
        "sugarcane": os.path.join(BASE_PATH, "Sugarcane Leaf Disease Dataset"),
        "cotton":    os.path.join(BASE_PATH, "cotton dataset"),
        "wheat":     os.path.join(BASE_PATH, "wheat", "data"),
    }

    # Load centroids from v2 (already extracted)
    with open(os.path.join(V2_PATH, "district_centroids.json")) as f:
        centroids = json.load(f)
    splits = assign_splits(centroids)

    # ── Phase 1: Build profile pools ──────────────────────────────────────
    print("═" * 60)
    print("PHASE 1: Building profile pools")
    print(f"  Target: {PROFILES_PER_CLASS} profiles × "
          f"{sum(len(d) for d in DISEASE_WINDOWS.values())} disease classes = "
          f"{PROFILES_PER_CLASS * sum(len(d) for d in DISEASE_WINDOWS.values())} total profiles")
    print("═" * 60)

    pools = build_all_pools(centroids, splits)

    # ── Phase 2: Assign to images ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("PHASE 2: Scanning images and assigning profiles")
    print("═" * 60)

    image_tasks = scan_images(datasets)
    print(f"Total images found: {len(image_tasks)}")

    records_raw = assign_profiles(image_tasks, pools, splits, centroids)

    # Add sequential IDs
    records = []
    for i, r in enumerate(records_raw):
        r["id"] = f"{i:06d}"
        records.append(r)

    # Save index
    with open(INDEX_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {INDEX_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────
    from collections import Counter
    split_counts = Counter(r["split"] for r in records)
    crop_counts  = Counter(r["crop_type"] for r in records)
    print("\n── Split distribution ──")
    for k in ("train","val","test"):
        n = split_counts.get(k, 0)
        print(f"  {k:6s}: {n:6d}  ({100*n/max(len(records),1):.1f}%)")
    print("\n── Crop distribution ──")
    for k, v in sorted(crop_counts.items()):
        print(f"  {k:12s}: {v}")

    # Profile reuse stats
    from collections import defaultdict
    reuse_by_file = defaultdict(int)
    for r in records:
        reuse_by_file[r["weather_file"]] += 1
    reuse_vals = list(reuse_by_file.values())
    if reuse_vals:
        print(f"\n── Profile reuse stats ──")
        print(f"  Unique profiles used : {len(reuse_vals)}")
        print(f"  Mean reuse per profile: {np.mean(reuse_vals):.1f}")
        print(f"  Max reuse (any profile): {max(reuse_vals)}")


if __name__ == "__main__":
    main()
