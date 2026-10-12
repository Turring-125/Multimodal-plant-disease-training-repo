"""
utils.py — v3
Utility functions for weather aggregation and validation.
Updated for 7-day window (was 30-day in v2).
"""
import numpy as np
import datetime


# ── Season encoding ───────────────────────────────────────────────────────────

def get_season_encoding(date_obj):
    """Cyclic month encoding so model gets continuous seasonal context."""
    month = date_obj.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return float(month_sin), float(month_cos)


# ── Hourly → daily aggregation ────────────────────────────────────────────────

def aggregate_hourly_to_daily(hourly_data, date_obj):
    """
    Aggregate 24-hour slices to 11 daily features.

    Parameters
    ----------
    hourly_data : dict
        Keys: 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
              'precipitation', 'wind_speed_10m', 'soil_temperature_0cm'
        Each value: list of 24 floats (may contain None for missing hours).
    date_obj : datetime.date
        The calendar date for the day (used for season encoding).

    Returns
    -------
    list of 11 floats:
        [temp_max, temp_min, temp_mean, rh_max, rh_mean, precip_sum,
         dewpoint_mean, wind_max, soil_temp, month_sin, month_cos]
    """
    sin_enc, cos_enc = get_season_encoding(date_obj)

    def safe_mean(data):
        clean = [x for x in data if x is not None]
        return float(np.mean(clean)) if clean else 0.0

    def safe_max(data):
        clean = [x for x in data if x is not None]
        return float(max(clean)) if clean else 0.0

    def safe_min(data):
        clean = [x for x in data if x is not None]
        return float(min(clean)) if clean else 0.0

    def safe_sum(data):
        clean = [x for x in data if x is not None]
        return float(sum(clean)) if clean else 0.0

    return [
        safe_max(hourly_data["temperature_2m"]),      # 0: temp_max
        safe_min(hourly_data["temperature_2m"]),      # 1: temp_min
        safe_mean(hourly_data["temperature_2m"]),     # 2: temp_mean
        safe_max(hourly_data["relative_humidity_2m"]),# 3: rh_max
        safe_mean(hourly_data["relative_humidity_2m"]),# 4: rh_mean
        safe_sum(hourly_data["precipitation"]),       # 5: precip_sum
        safe_mean(hourly_data["dew_point_2m"]),       # 6: dewpoint_mean
        safe_max(hourly_data["wind_speed_10m"]),      # 7: wind_max
        safe_mean(hourly_data["soil_temperature_0cm"]),# 8: soil_temp
        sin_enc,                                       # 9: month_sin
        cos_enc,                                       # 10: month_cos
    ]


# ── Favorable-condition validation (7-day window) ─────────────────────────────

# NOTE v3: Window is 7 days. Thresholds require ≥3 favorable days (was ≥7/30 in v2).
# This is proportionally equivalent (3/7 ≈ 43% vs 7/30 ≈ 23%, slightly stricter
# per day but realistic for a short window).

FAVORABLE_THRESHOLDS = {
    # --- WHEAT ---
    "wheat": {
        "Yellow Rust":        lambda df: (df[:, 2] >= 8)  & (df[:, 2] <= 18) & (df[:, 4] > 75),
        "Brown Rust":         lambda df: (df[:, 2] >= 14) & (df[:, 2] <= 25) & (df[:, 4] > 65),
        "Black Rust":         lambda df: (df[:, 2] >= 18) & (df[:, 2] <= 28) & (df[:, 4] > 65),
        "Mildew":             lambda df: (df[:, 2] >= 8)  & (df[:, 2] <= 20) & (df[:, 4] > 55),
        "Septoria":           lambda df: (df[:, 2] >= 12) & (df[:, 2] <= 22) & (df[:, 5] > 2.0),
        "Tan Spot":           lambda df: (df[:, 2] >= 18) & (df[:, 2] <= 30) & (df[:, 4] > 50),
        "Fusarium Head Blight": lambda df: (df[:, 2] >= 22) & (df[:, 2] <= 32) & (df[:, 4] > 85),
        "Smut":               lambda df: (df[:, 2] >= 8)  & (df[:, 2] <= 18) & (df[:, 4] > 55),
        "Aphid":              lambda df: (df[:, 2] >= 12) & (df[:, 2] <= 27),
        "Mite":               lambda df: (df[:, 2] >= 22) & (df[:, 4] < 55),
        "Blast":              lambda df: (df[:, 2] >= 20) & (df[:, 2] <= 30) & (df[:, 4] > 82),
        "Common Root Rot":    lambda df: (df[:, 8] >= 12),
        "Stem Fly":           lambda df: (df[:, 2] >= 12) & (df[:, 2] <= 27),
        "Leaf Blight":        lambda df: (df[:, 2] >= 22) & (df[:, 4] > 65),
        "Healthy":            lambda df: np.ones(len(df), dtype=bool),
    },
    # --- RICE ---
    "rice": {
        "Leaf Blast":              lambda df: (df[:, 2] >= 22) & (df[:, 2] <= 30) & (df[:, 4] > 85),
        "Bacterial Leaf Blight":   lambda df: (df[:, 2] >= 25) & (df[:, 2] <= 35) & (df[:, 4] > 70),
        "Brown Spot":              lambda df: (df[:, 2] >= 24) & (df[:, 2] <= 32),
        "Sheath Blight":           lambda df: (df[:, 2] >= 26) & (df[:, 2] <= 34) & (df[:, 4] > 80),
        "Leaf Scald":              lambda df: (df[:, 2] >= 22) & (df[:, 2] <= 32),
        "Healthy Rice Leaf":       lambda df: np.ones(len(df), dtype=bool),
    },
    # --- SUGARCANE ---
    "sugarcane": {
        "Redrot":         lambda df: (df[:, 2] >= 24) & (df[:, 2] <= 35) & (df[:, 4] > 75),
        "Mosaic":         lambda df: (df[:, 2] >= 18) & (df[:, 2] <= 32),
        "Rust":           lambda df: (df[:, 2] >= 16) & (df[:, 2] <= 26) & (df[:, 4] > 75),
        "Yellow Disease": lambda df: (df[:, 2] <= 22),
        "Healthy":        lambda df: np.ones(len(df), dtype=bool),
    },
    # --- COTTON ---
    "cotton": {
        "Bacterial Blight":   lambda df: (df[:, 2] >= 28) & (df[:, 2] <= 38) & (df[:, 4] > 70),
        "Curl Virus":         lambda df: (df[:, 2] >= 24) & (df[:, 2] <= 37) & (df[:, 4] < 72),
        "Leaf Reddening":     lambda df: (df[:, 2] >= 32),
        "Leaf Variegation":   lambda df: (df[:, 2] >= 20) & (df[:, 2] <= 35),
        "Leaf Hopper Jassids":lambda df: (df[:, 2] >= 24) & (df[:, 2] <= 37) & (df[:, 4] < 72),
        "Herbicide Damage":   lambda df: np.ones(len(df), dtype=bool),
        "Healthy Leaf":       lambda df: np.ones(len(df), dtype=bool),
    },
}

# Minimum favorable days required (out of 7)
MIN_FAVORABLE_DAYS = 3


def validate_weather_window(crop, disease, daily_features):
    """
    Returns True if >= MIN_FAVORABLE_DAYS out of 7 are agronomically
    favorable for the given crop/disease combination.

    Parameters
    ----------
    crop : str
        One of 'wheat', 'rice', 'sugarcane', 'cotton'.
    disease : str
        Disease label matching FAVORABLE_THRESHOLDS keys.
    daily_features : list of lists
        Shape [7, 11] — output of aggregate_hourly_to_daily for each day.

    Returns
    -------
    bool
    """
    df = np.array(daily_features, dtype=np.float32)  # [7, 11]

    crop_thresholds = FAVORABLE_THRESHOLDS.get(crop)
    if crop_thresholds is None:
        return True  # unknown crop → pass

    # Find matching disease key (case-insensitive partial match)
    fn = None
    for key, func in crop_thresholds.items():
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            fn = func
            break

    if fn is None:
        return True  # unknown disease → pass

    mask = fn(df)
    return int(np.sum(mask)) >= MIN_FAVORABLE_DAYS
