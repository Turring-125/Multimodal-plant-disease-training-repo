import numpy as np
import datetime

def get_season_encoding(date_obj):
    month = date_obj.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos

def aggregate_hourly_to_daily(hourly_data, date_obj):
    # hourly_data is a dict with keys like 'temperature_2m', etc.
    # Each value is a list of 24 values.
    
    sin_enc, cos_enc = get_season_encoding(date_obj)
    
    def safe_mean(data):
        clean = [x for x in data if x is not None]
        return np.mean(clean) if clean else 0.0

    def safe_max(data):
        clean = [x for x in data if x is not None]
        return max(clean) if clean else 0.0

    def safe_min(data):
        clean = [x for x in data if x is not None]
        return min(clean) if clean else 0.0

    def safe_sum(data):
        clean = [x for x in data if x is not None]
        return sum(clean) if clean else 0.0
    
    daily = {
        "temp_max":      safe_max(hourly_data["temperature_2m"]),
        "temp_min":      safe_min(hourly_data["temperature_2m"]),
        "temp_mean":     safe_mean(hourly_data["temperature_2m"]),
        "rh_max":        safe_max(hourly_data["relative_humidity_2m"]),
        "rh_mean":       safe_mean(hourly_data["relative_humidity_2m"]),
        "precip_sum":    safe_sum(hourly_data["precipitation"]),
        "dewpoint_mean": safe_mean(hourly_data["dew_point_2m"]),
        "wind_max":      safe_max(hourly_data["wind_speed_10m"]),
        "soil_temp":     safe_mean(hourly_data["soil_temperature_0cm"]),
        "month_sin":     sin_enc,
        "month_cos":     cos_enc,
    }
    
    # Return as a list/array in the correct order (11 columns)
    return [
        daily["temp_max"], daily["temp_min"], daily["temp_mean"],
        daily["rh_max"], daily["rh_mean"], daily["precip_sum"],
        daily["dewpoint_mean"], daily["wind_max"], daily["soil_temp"],
        daily["month_sin"], daily["month_cos"]
    ]

def validate_weather_window(crop, disease, daily_features):
    """
    Validates if a 30-day weather sequence contains enough days 
    matching the favorable conditions for a disease.
    """
    df = np.array(daily_features)
    # Features: [0:t_max, 1:t_min, 2:t_mean, 3:rh_max, 4:rh_mean, 5:precip_sum, 6:dew_mean, 7:wind_max, 8:soil_temp]
    
    # Define thresholds based on INSTRUCTIONS_v2.md
    if crop == "wheat":
        if "Yellow Rust" in disease:
            mask = (8 <= df[:, 2]) & (df[:, 2] <= 18) & (df[:, 4] > 75)
            return np.sum(mask) >= 7 # At least 1 week of ideal conditions
        if "Brown Rust" in disease:
            mask = (15 <= df[:, 2]) & (df[:, 2] <= 25) & (df[:, 4] > 65)
            return np.sum(mask) >= 7
        if "Black Rust" in disease:
            mask = (18 <= df[:, 2]) & (df[:, 2] <= 28) & (df[:, 4] > 65)
            return np.sum(mask) >= 7
        if "Blast" in disease:
            mask = (22 <= df[:, 2]) & (df[:, 2] <= 30) & (df[:, 4] > 85)
            return np.sum(mask) >= 5
    
    if crop == "rice":
        if "Leaf Blast" in disease:
            mask = (22 <= df[:, 2]) & (df[:, 2] <= 30) & (df[:, 4] > 85)
            return np.sum(mask) >= 7
        if "Bacterial Leaf Blight" in disease:
            mask = (25 <= df[:, 2]) & (df[:, 2] <= 35) & (df[:, 4] > 75)
            return np.sum(mask) >= 7
        if "Brown Spot" in disease:
            mask = (25 <= df[:, 2]) & (df[:, 2] <= 32) & (df[:, 4] < 80)
            return np.sum(mask) >= 7
            
    if crop == "sugarcane":
        if "Redrot" in disease:
            mask = (25 <= df[:, 2]) & (df[:, 2] <= 35) & (df[:, 4] > 75)
            return np.sum(mask) >= 7
        if "Rust" in disease:
            mask = (18 <= df[:, 2]) & (df[:, 2] <= 25) & (df[:, 4] > 75)
            return np.sum(mask) >= 7
            
    if crop == "cotton":
        if "Bacterial Blight" in disease:
            mask = (25 <= df[:, 2]) & (df[:, 2] <= 35) & (df[:, 4] > 75)
            return np.sum(mask) >= 7
        if "Curl Virus" in disease:
            mask = (25 <= df[:, 2]) & (df[:, 2] <= 35) & (df[:, 4] < 70)
            return np.sum(mask) >= 7
            
    return True
