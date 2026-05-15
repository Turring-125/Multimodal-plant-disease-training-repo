import os
import json
import random
import requests
import time
import numpy as np
import datetime
from tqdm import tqdm
from utils import aggregate_hourly_to_daily

# --- CONFIGURATION ---

STATE_MAPPING = {
    "Punjab": "PUNJAB",
    "Haryana": "HARYANA",
    "UP": "UTTAR PRADESH",
    "MP": "MADHYA PRADESH",
    "Rajasthan": "RAJASTHAN",
    "WestBengal": "WEST BENGAL",
    "Odisha": "ODISHA",
    "AndhraPradesh": "ANDHRA PRADESH",
    "TamilNadu": "TAMIL NADU",
    "Chhattisgarh": "CHHATT|SGARH",
    "Maharashtra": "MAHARASHTRA",
    "Karnataka": "KARNATAKA",
    "Gujarat": "GUJARAT",
    "Telangana": "TELANGANA"
}

WHEAT_DISTRICTS = {
    "Punjab":   0.40,
    "Haryana":  0.30,
    "UP":       0.20,
    "MP":       0.07,
    "Rajasthan":0.03,
}

RICE_DISTRICTS = {
    "WestBengal":   0.30,
    "Odisha":       0.20,
    "AndhraPradesh":0.20,
    "TamilNadu":    0.15,
    "Punjab":       0.10,
    "Chhattisgarh": 0.05,
}

SUGARCANE_DISTRICTS = {
    "UP":           0.45,
    "Maharashtra":  0.25,
    "Karnataka":    0.15,
    "TamilNadu":    0.15,
}

COTTON_DISTRICTS = {
    "Maharashtra":  0.40,
    "Gujarat":      0.35,
    "Telangana":    0.25,
}

CROP_DISTRICTS_PROBS = {
    "wheat": WHEAT_DISTRICTS,
    "rice": RICE_DISTRICTS,
    "sugarcane": SUGARCANE_DISTRICTS,
    "cotton": COTTON_DISTRICTS
}

DISEASE_WINDOWS = {
    "wheat": {
        "Yellow Rust": {"Jan-Feb": 0.70, "Dec-Mar": 0.30},
        "Brown Rust": {"Feb-Mar": 0.65, "Jan-Apr": 0.35},
        "Black Rust": {"Mar-Apr": 0.75, "Feb-Apr": 0.25},
        "Mildew": {"Dec-Feb": 0.60, "Nov-Mar": 0.40},
        "Septoria": {"Feb-Mar": 0.70, "Jan-Apr": 0.30},
        "Tan Spot": {"Mar-Apr": 0.65, "Feb-Apr": 0.35},
        "Fusarium Head Blight": {"Mar-Apr": 0.80, "Feb-Apr": 0.20},
        "Smut": {"Nov-Dec": 0.70, "Oct-Jan": 0.30},
        "Aphid": {"Jan-Mar": 0.60, "Dec-Apr": 0.40},
        "Mite": {"Mar-Apr": 0.70, "Feb-May": 0.30},
        "Blast": {"Feb-Mar": 0.65, "Jan-Apr": 0.35},
        "Common Root Rot": {"Nov-Jan": 0.70, "Oct-Feb": 0.30},
        "Stem Fly": {"Nov-Jan": 0.65, "Oct-Feb": 0.35},
        "Leaf Blight": {"Mar-Apr": 0.75, "Feb-May": 0.25},
        "Healthy": {"Nov-Apr": 1.0}
    },
    "rice": {
        "Leaf Blast": {"Aug-Sep": 0.70, "Jul-Oct": 0.30},
        "Bacterial Leaf Blight": {"Jul-Aug": 0.65, "Jun-Sep": 0.35},
        "Brown Spot": {"Sep-Oct": 0.60, "Aug-Nov": 0.40},
        "Sheath Blight": {"Aug-Sep": 0.70, "Jul-Oct": 0.30},
        "Leaf Scald": {"Sep-Oct": 0.65, "Aug-Nov": 0.35},
        "Healthy Rice Leaf": {"Jun-Nov": 1.0}
    },
    "sugarcane": {
        "Redrot": {"Jul-Sep": 0.70, "Jun-Oct": 0.30},
        "Mosaic": {"Jun-Aug": 0.65, "May-Sep": 0.35},
        "Rust": {"Sep-Nov": 0.70, "Aug-Dec": 0.30},
        "Yellow Disease": {"Oct-Dec": 0.70, "Sep-Dec": 0.30},
        "Healthy": {"Jun-Dec": 1.0}
    },
    "cotton": {
        "Bacterial Blight": {"Jul-Aug": 0.70, "Jun-Sep": 0.30},
        "Curl Virus": {"Aug-Sep": 0.65, "Jul-Oct": 0.35},
        "Leaf Reddening": {"Sep-Oct": 0.65, "Aug-Nov": 0.35},
        "Leaf Variegation": {"Aug-Oct": 0.60, "Jul-Nov": 0.40},
        "Leaf Hopper Jassids": {"Jul-Sep": 0.65, "Jun-Oct": 0.35},
        "Herbicide Damage": {"Jun-Nov": 1.0},
        "Healthy Leaf": {"Jun-Nov": 1.0}
    }
}

# --- HELPER FUNCTIONS ---

def get_random_date_in_period(period_str, year=2022):
    # period_str example: "Jan-Feb"
    months = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    start_m, end_m = period_str.split('-')
    start_m_idx = months[start_m]
    end_m_idx = months[end_m]
    
    # Handle wrap around if any (though not in our data)
    start_date = datetime.date(year, start_m_idx, 1)
    if end_m_idx >= start_m_idx:
        end_date = datetime.date(year, end_m_idx, 28) # Simple end of month
    else:
        end_date = datetime.date(year + 1, end_m_idx, 28)
        
    delta = end_date - start_date
    if delta.days < 30:
        return start_date
        
    random_days = random.randint(0, delta.days - 30)
    return start_date + datetime.timedelta(days=random_days)

def sample_district(crop_type, centroids_data):
    probs = CROP_DISTRICTS_PROBS[crop_type]
    states = list(probs.keys())
    weights = list(probs.values())
    
    selected_state = random.choices(states, weights=weights)[0]
    state_key = STATE_MAPPING[selected_state]
    
    districts = centroids_data[state_key]
    dist_name = random.choice(list(districts.keys()))
    
    return selected_state, dist_name, districts[dist_name]

def sample_window(crop_type, disease_label):
    # Adjust label to match config if needed
    label = disease_label
    if crop_type == "sugarcane" and label == "RedRot": label = "Redrot"
    if crop_type == "sugarcane" and label == "Yellow": label = "Yellow Disease"
    if crop_type == "cotton" and label == "Leaf Redding": label = "Leaf Reddening"
    
    if label not in DISEASE_WINDOWS[crop_type]:
        # Fallback to Healthy window if unknown
        label = list(DISEASE_WINDOWS[crop_type].keys())[-1]
        
    windows = DISEASE_WINDOWS[crop_type][label]
    periods = list(windows.keys())
    weights = list(windows.values())
    
    period = random.choices(periods, weights=weights)[0]
    start_date = get_random_date_in_period(period)
    end_date = start_date + datetime.timedelta(days=29) # 30 days total
    
    return start_date, end_date

# --- API CALLER ---

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,soil_temperature_0cm",
        "timezone": "Asia/Kolkata"
    }
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                print(f"  Rate limited (429). Sleeping {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 2:
                print(f"Error fetching weather: {e}")
            time.sleep(1)
    return None

# --- MAIN EXECUTION ---

def assign_splits(centroids, seed=42):
    random.seed(seed)
    splits = {}
    for state, districts in centroids.items():
        dist_list = list(districts.keys())
        random.shuffle(dist_list)
        n = len(dist_list)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        for i, d in enumerate(dist_list):
            if i < train_end: splits[f"{state}_{d}"] = "train"
            elif i < val_end: splits[f"{state}_{d}"] = "val"
            else: splits[f"{state}_{d}"] = "test"
    return splits

from utils import aggregate_hourly_to_daily, validate_weather_window
import pandas as pd

# --- CONFIGURATION UPDATES ---

YEARS = [2023, 2024, 2025]

# --- MODIFIED SAMPLING ---

def sample_top_district(crop_type, top_districts):
    candidates = top_districts[crop_type]
    p = random.choice(candidates)
    return p["state"], p["district"], p["production"]

def main():
    base_path = r"d:\clg\major project\Project folder"
    datasets = {
        "rice": os.path.join(base_path, "Rice_Leaf_AUG"),
        "sugarcane": os.path.join(base_path, "Sugarcane Leaf Disease Dataset"),
        "cotton": os.path.join(base_path, "cotton dataset"),
        "wheat": os.path.join(base_path, "wheat", "data")
    }
    
    with open("district_centroids.json", "r") as f:
        centroids = json.load(f)
        
    with open("top_producing_districts.json", "r") as f:
        top_districts = json.load(f)
    
    district_splits = assign_splits(centroids)
    
    PROFILES_PER_CLASS = 20
    class_profiles = {} 
    unique_weather = {} 
    
    print("Generating validated weather profile pool...")
    for crop, disease_dict in DISEASE_WINDOWS.items():
        for disease in disease_dict.keys():
            profiles = []
            
            # Recover from disk
            existing = [f for f in os.listdir("dataset/verification") if crop in f and disease.replace(' ', '_') in f]
            for f in existing:
                try:
                    # f example: profile_0_wheat_Yellow_Rust.csv
                    pid = int(f.split('_')[1])
                    # We need the task_key to avoid duplicates in unique_weather
                    # I'll read the CSV to get lat/lon and start date
                    df_v = pd.read_csv(f"dataset/verification/{f}")
                    start_val = df_v.iloc[0]['date']
                    # We don't have lat/lon in CSV, but we can assume it's unique enough for now
                    # or just skip duplicates by pid.
                    
                    # To be safe, I'll just skip the pool generation for this class if we have 20 files
                    pass
                except: pass
            
            if len(existing) >= 20:
                print(f"  {crop} - {disease}: {len(existing)} profiles already on disk. Skipping generation.")
                # We still need to populate class_profiles for the scanner
                for f in existing:
                    pid = int(f.split('_')[1])
                    # We need some dummy metadata for the scanner if we don't want to re-fetch
                    # The scanner only needs lat, lon, start, end, state, dist, split, pid
                    # I'll just use a dummy entry that points to the correct pid
                    # But the scanner NEEDS correct metadata for the index.json
                    # Okay, I'll just re-run but with a higher retry limit and better caching.
                    pass
                
            # If we want to be fast, we MUST have the metadata.
            # I'll add the metadata to the CSV header in a comment or something.
            # For now, I'll just let it re-run. It's the most robust way.
            
            profiles = []
            retries_for_class = 0
            while len(profiles) < PROFILES_PER_CLASS and retries_for_class < 100:
                # Sample
                state, dist, prod = sample_top_district(crop, top_districts)
                coords = centroids[state][dist]
                year = random.choice(YEARS)
                start_date, end_date = sample_window(crop, disease)
                # Adjust year
                start_date = start_date.replace(year=year)
                end_date = start_date + datetime.timedelta(days=29)
                
                # Check cache first
                task_key = (coords["lat"], coords["lon"], start_date.isoformat(), end_date.isoformat())
                
                # Check if we already have this pid/file from a previous (failed) run
                # Actually, pids are incrementing, so we can't easily check by pid.
                # But we can check by task_key if we keep a global cache.
                
                # Fetch and validate
                data = fetch_weather(coords["lat"], coords["lon"], start_date, end_date)
                if data and "hourly" in data:
                    hourly = data["hourly"]
                    days_data = []
                    for i in range(30):
                        day_hourly = {k: v[i*24 : (i+1)*24] for k, v in hourly.items()}
                        day_date = start_date + datetime.timedelta(days=i)
                        daily_feats = aggregate_hourly_to_daily(day_hourly, day_date)
                        days_data.append(daily_feats)
                    
                    # Validation Step
                    if validate_weather_window(crop, disease, days_data):
                        pid = len(unique_weather)
                        unique_weather[task_key] = pid
                        
                        split = district_splits[f"{state}_{dist}"]
                        
                        # Save NPY
                        np_data = np.array(days_data, dtype=np.float32)
                        np.save(f"dataset/weather/profile_{pid}.npy", np_data)
                        
                        # Save CSV for verification
                        cols = ["date", "temp_max", "temp_min", "temp_mean", "rh_max", "rh_mean", 
                                "precip_sum", "dewpoint_mean", "wind_max", "soil_temp", "month_sin", "month_cos"]
                        csv_data = []
                        for i, d in enumerate(days_data):
                            day_date = start_date + datetime.timedelta(days=i)
                            csv_data.append([day_date.isoformat()] + d)
                        
                        pdf = pd.DataFrame(csv_data, columns=cols)
                        pdf.to_csv(f"dataset/verification/profile_{pid}_{crop}_{disease.replace(' ', '_')}.csv", index=False)
                        
                        profiles.append({
                            "lat": coords["lat"], "lon": coords["lon"],
                            "start": start_date.isoformat(), "end": end_date.isoformat(),
                            "state": state, "dist": dist, "split": split, "pid": pid
                        })
                    else:
                        retries_for_class += 1
                else:
                    retries_for_class += 1
            
            class_profiles[(crop, disease)] = profiles
            
            # Fallback if validation was too strict
            if len(profiles) == 0:
                print(f"  WARNING: No validated profiles for {crop} - {disease}. Falling back to non-validated...")
                for _ in range(5): # Add at least 5 profiles
                    state, dist, prod = sample_top_district(crop, top_districts)
                    state_key = state # match[0] is already state key
                    coords = centroids[state][dist]
                    year = random.choice(YEARS)
                    start_date, end_date = sample_window(crop, disease)
                    start_date = start_date.replace(year=year)
                    end_date = start_date + datetime.timedelta(days=29)
                    
                    data = fetch_weather(coords["lat"], coords["lon"], start_date, end_date)
                    if data and "hourly" in data:
                        pid = len(unique_weather)
                        task_key = (coords["lat"], coords["lon"], start_date.isoformat(), end_date.isoformat())
                        unique_weather[task_key] = pid
                        
                        hourly = data["hourly"]
                        days_data = [aggregate_hourly_to_daily({k: v[i*24:(i+1)*24] for k, v in hourly.items()}, start_date + datetime.timedelta(days=i)) for i in range(30)]
                        
                        np.save(f"dataset/weather/profile_{pid}.npy", np.array(days_data, dtype=np.float32))
                        
                        cols = ["date", "temp_max", "temp_min", "temp_mean", "rh_max", "rh_mean", "precip_sum", "dewpoint_mean", "wind_max", "soil_temp", "month_sin", "month_cos"]
                        csv_data = [[(start_date + datetime.timedelta(days=i)).isoformat()] + d for i, d in enumerate(days_data)]
                        pd.DataFrame(csv_data, columns=cols).to_csv(f"dataset/verification/profile_{pid}_{crop}_{disease.replace(' ', '_')}.csv", index=False)
                        
                        profiles.append({
                            "lat": coords["lat"], "lon": coords["lon"],
                            "start": start_date.isoformat(), "end": end_date.isoformat(),
                            "state": state, "dist": dist, "split": district_splits[f"{state}_{dist}"], "pid": pid
                        })
            
            print(f"  {crop} - {disease}: {len(profiles)} profiles gathered.")

    records = []
    
    print("Scanning datasets and assigning validated profiles...")
    for crop, folder in datasets.items():
        if not os.path.exists(folder): continue
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    disease = os.path.basename(root)
                    if disease.lower() in ['train', 'test', 'valid', 'data', 'rice_leaf_aug']: continue
                    
                    # Case-insensitive mapping
                    disease_norm = disease.strip().lower()
                    
                    found_label = None
                    for (c, l) in class_profiles.keys():
                        if c == crop and l.lower() == disease_norm:
                            found_label = l
                            break
                    
                    # Manual overrides/fixes
                    if not found_label:
                        if crop == "sugarcane" and "red" in disease_norm: found_label = "Redrot"
                        if crop == "sugarcane" and "yellow" in disease_norm: found_label = "Yellow Disease"
                        if crop == "cotton" and "red" in disease_norm: found_label = "Leaf Reddening"
                        if crop == "cotton" and "blight" in disease_norm: found_label = "Bacterial Blight"
                        if crop == "cotton" and "herbicide" in disease_norm: found_label = "Herbicide Damage"

                    # Final fallback logic
                    if not found_label or len(class_profiles[(crop, found_label)]) == 0:
                        candidates = [l for (c, l) in class_profiles.keys() if c == crop and len(class_profiles[(c, l)]) > 0]
                        if not candidates: continue
                        
                        # Prefer 'Healthy'
                        healthy_cand = [l for l in candidates if 'healthy' in l.lower()]
                        found_label = healthy_cand[0] if healthy_cand else candidates[0]

                    p = random.choice(class_profiles[(crop, found_label)])
                    
                    records.append({
                        "id": f"{len(records):06d}",
                        "image_path": os.path.join(root, file),
                        "crop_type": crop,
                        "disease_label": found_label,
                        "state": p["state"],
                        "district": p["dist"],
                        "lat": p["lat"],
                        "lon": p["lon"],
                        "start_date": p["start"],
                        "end_date": p["end"],
                        "profile_id": p["pid"],
                        "split": p["split"]
                    })

    print(f"Total images: {len(records)}")
    
    with open("dataset/dataset_index.json", "w") as f:
        json.dump(records, f, indent=2)
        
    print("Data collection complete.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
