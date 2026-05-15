# Data Collection and Processing Documentation

This document outlines the pipeline used to create the multimodal environmental-image dataset for plant disease detection.

## 1. Geographical Mapping & Centroids
The foundation of the environmental data is the mapping of Indian districts to their geometric centers (centroids).
- **Source**: `DISTRICT_BOUNDARY.shp` (Indian District Shapefiles).
- **Transformation**: The shapefile used a **Lambert Conformal Conic (LCC)** projection. We transformed these coordinates into **WGS84 (Latitude/Longitude)** using the `pyproj` library to ensure compatibility with global weather APIs.
- **Extraction**: We used the `shapely` library to calculate the centroid of each district polygon, resulting in `district_centroids.json`.

## 2. District Targeting (ICRISAT Integration)
To ensure the weather data represents major agricultural regions, we integrated production data from the **ICRISAT District Level Database**.
- **Selection Logic**: We identified the **top 20 producing districts** for each major crop (Wheat, Rice, Sugarcane, Cotton).
- **Sampling**: When generating a weather profile for a specific crop, the system randomly samples from these top-producing districts rather than using a uniform national distribution.

## 3. Environmental Data Collection (Open-Meteo)
Historical weather data was retrieved using the **Open-Meteo Historical Archive API**.
- **Temporal Range**: Data was sampled from the years **2023, 2024, and 2025**.
- **Data Resolution**: Hourly data was fetched for each location and time window.
- **Feature Engineering**: Hourly data was aggregated into **11 daily features**:
    - Temperature (Max, Min, Mean)
    - Relative Humidity (Max, Mean)
    - Precipitation (Daily Sum)
    - Dew Point (Mean)
    - Wind Speed (Max)
    - Soil Temperature (0cm)
    - Cyclical Season Encoding (Sin/Cos of Month)

## 4. Disease-Weather Mapping & Validation
A critical part of this pipeline is ensuring the weather "matches" the disease. We implemented an **Active Validation Engine** to filter profiles.
- **Time Windows**: For each disease, we defined a "Favorable Window" based on agricultural calendars (e.g., Jan-Feb for Wheat Yellow Rust).
- **Favorable Conditions**: Every 30-day weather profile was verified against specific temperature and humidity thresholds.
- **The 7-Day Rule**: A profile is only accepted if it contains at least **7 days** of ideal conditions for that specific disease. This ensures the model learns the actual environmental triggers of the infection.
- **Fallback**: If a highly specific condition cannot be found after 100 retries, the system falls back to a standard seasonal window to maintain dataset coverage.

## 5. Dataset Indexing & Splitting
The final output is a master index (`dataset_index.json`) that links every image to its environmental context.
- **Total Samples**: 22,641 images across 4 crops.
- **Data Partitioning**: We implemented a **District-Based Split** (approx. 70/15/15):
    - All images from a specific district are kept together in either the Train, Val, or Test set.
    - This prevents "Geographic Data Leakage," ensuring the model generalizes to new regions rather than just memorizing local weather patterns.
- **Storage**:
    - `.npy` files: Compressed NumPy arrays for fast GPU loading during training.
    - `.csv` files: Human-readable verification files located in `projectv2/dataset/verification/`.

## 6. Verification Tools
To ensure data integrity, the following scripts were used:
- `verify_data.py`: Validates the shape of all `.npy` files and the consistency of the master index.
- `analyze_icrisat.py`: Ranks districts by production to drive the sampling engine.
