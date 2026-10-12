# Multimodal Crop Disease Detection — Pipeline Instructions v2
### For: Agent + Developer Reference
### Crops: Wheat · Rice · Sugarcane · Cotton
### Framing: Context-Aware Disease Classification Using Weakly Aligned Environmental Priors

> **⚠️ IMPORTANT — READ BEFORE ANYTHING ELSE**
> The weather data in this pipeline is **NOT** matched to the actual field where the image was taken.
> It is **synthetically assigned** based on agronomic priors (which conditions favor which disease).
> This must be explicitly disclosed in the paper as **"weakly aligned environmental conditioning"**.
> Do NOT claim real-world paired data. This is a limitation AND a contribution — you are building an
> environmental context augmentation framework, not a sensor-fused field system.

---

## PART 1 — ENVIRONMENTAL PARAMETERS TO DOWNLOAD FROM OPEN-METEO

Use the **Historical Archive API** only:
```
https://archive-api.open-meteo.com/v1/archive
```
NOT the forecast endpoint. This gives real reanalysis data for past seasons.

### V1 Core Features (Start here — build baseline first)

These 8 daily features are reliable, disease-relevant, and low-noise:

| Variable | Open-Meteo Field | Reliability | Disease Relevance |
|---|---|---|---|
| Temperature (2m) | `temperature_2m` | Excellent | Fungal/bacterial growth thresholds |
| Relative Humidity (2m) | `relative_humidity_2m` | Excellent | Spore germination trigger |
| Dewpoint (2m) | `dew_point_2m` | Good | Leaf wetness proxy |
| Precipitation | `precipitation` | Excellent | Splash dispersal of spores |
| Rain | `rain` | Good | Separates rainfall from dew/fog |
| Wind Speed (10m) | `wind_speed_10m` | Good | Airborne spore dispersal |
| Soil Temperature (0cm) | `soil_temperature_0cm` | Decent | Soilborne pathogen activity |
| **Season Encoding** | *(computed, not API)* | — | Disease timing context |

**Season encoding** — add 2 computed columns per day:
```python
import numpy as np
month = date.month  # 1–12
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```
This gives the model cyclical temporal context without hard-coding seasons. Add these as columns 9 and 10.

**V1 final shape per sample**: `[30 days × 10 features]`

---

### V2 Optional Features (Add only after V1 baseline works)

| Variable | Open-Meteo Field | Caveat |
|---|---|---|
| Vapour Pressure Deficit | `vapour_pressure_deficit` | Derived metric; regionally variable |
| Soil Moisture (0–1 cm) | `soil_moisture_0_to_1cm` | Reanalysis estimate, NOT true field soil |
| Soil Moisture (1–3 cm) | `soil_moisture_1_to_3cm` | Spatially noisy |
| ET₀ Reference | `et0_fao_evapotranspiration` | Correlated with temp+humidity, adds dimensionality |

> **Why defer these**: Soil moisture variables from Open-Meteo are **weather-model-derived approximations**, not sensor measurements. They are spatially coarse for Indian district centroids and may add noise rather than signal. Validate on V1 first.

---

### Aggregation (hourly → daily)

Request **hourly** data, aggregate to **daily** features:

```python
daily_features = {
    "temp_max":     hourly["temperature_2m"].max(),
    "temp_min":     hourly["temperature_2m"].min(),
    "temp_mean":    hourly["temperature_2m"].mean(),
    "rh_max":       hourly["relative_humidity_2m"].max(),
    "rh_mean":      hourly["relative_humidity_2m"].mean(),
    "precip_sum":   hourly["precipitation"].sum(),
    "dewpoint_mean":hourly["dew_point_2m"].mean(),
    "wind_max":     hourly["wind_speed_10m"].max(),
    "soil_temp":    hourly["soil_temperature_0cm"].mean(),
    "month_sin":    sin_encoding,   # same value all 24hrs of that day
    "month_cos":    cos_encoding,
}
# Drop "rain" — redundant with precipitation for Indian monsoon context
```

> **Note**: `temp_max` and `temp_min` are kept separate because many fungal diseases have both a temperature minimum threshold (activation) and maximum threshold (inhibition). Collapsing to mean loses this signal.

---

## PART 2 — CROP CALENDARS & DOWNLOAD WINDOWS

Download weather for the **full disease-prone growth window**, not just sowing. Each image sample will use a **30-day slice** within this window, centered on the disease-favorable period.

### Wheat (Rabi crop)
**Primary states**: Punjab, Haryana, western UP, northern MP, Rajasthan (Sri Ganganagar, Hanumangarh)

| Growth Phase | Months | Active Diseases |
|---|---|---|
| Sowing | Nov | Smut, Common Root Rot, Stem Fly |
| Vegetative | Dec – Jan | Mildew, Aphid, Mite |
| Reproductive / Booting | Feb – Mar | Yellow Rust, Brown Rust, Septoria, Tan Spot, Blast |
| Grain Fill / Heading | Mar – Apr | Black Rust, Fusarium Head Blight |
| Harvest | Apr | — |

**Full download window**: Nov 1 → Apr 15
**Training window for sampling**: Dec 1 → Apr 10 (exclude very early sowing noise)

---

### Rice (Kharif crop)
**Primary states**: West Bengal, Odisha, Andhra Pradesh, Tamil Nadu, Punjab (kharif), Chhattisgarh

| Growth Phase | Months | Active Diseases |
|---|---|---|
| Nursery / Transplant | Jun – Jul | — |
| Tillering | Jul – Aug | Bacterial Leaf Blight, Brown Spot |
| Booting | Aug – Sep | Leaf Blast, Sheath Blight |
| Heading / Ripening | Sep – Oct | Leaf Scald, Brown Spot |
| Harvest | Oct – Nov | — |

**Full download window**: Jun 15 → Nov 15
**Training window for sampling**: Jul 15 → Nov 1

---

### Sugarcane (Annual/Ratoon)
**Primary states**: Western UP (Meerut, Muzaffarnagar), Maharashtra (Pune, Kolhapur, Nashik), Karnataka (Belagavi), Tamil Nadu (Erode)

| Growth Phase | Months | Active Diseases |
|---|---|---|
| Planting | Feb – Mar | — |
| Grand Growth | Jun – Sep | Mosaic, Redrot, Rust |
| Maturation | Oct – Dec | Yellow Disease, Rust |
| Harvest | Jan – Feb | — |

**Full download window**: Jun 1 → Dec 31
**Training window for sampling**: Jun 1 → Dec 20

---

### Cotton (Kharif crop)
**Primary states**: Maharashtra Vidarbha (Nagpur, Amravati, Yavatmal), Gujarat (Saurashtra), Telangana (Warangal, Adilabad)

| Growth Phase | Months | Active Diseases |
|---|---|---|
| Sowing | May – Jun | — |
| Vegetative | Jul | Bacterial Blight, Leaf Hopper Jassids |
| Boll Development | Aug – Sep | Curl Virus, Leaf Reddening |
| Boll Opening | Oct | Leaf Variegation |
| Harvest | Nov – Dec | — |

**Full download window**: Jul 1 → Nov 30
**Training window for sampling**: Jul 1 → Nov 20

---

## PART 3 — DISTRICTS & PROBABILISTIC SAMPLING

Only use districts where the crop is a **dominant cultivar** (>20% of district agricultural area).
Source: ICRISAT District-level Database or state agriculture department reports.

**Agent instruction**: Extract **centroid lat/lon** from shapefile for each district. Use centroid for API calls. Do NOT use bounding box polygons.

---

### District Probability Distributions (CRITICAL — do NOT sample uniformly)

Sampling must reflect actual agricultural dominance of each region. Skewed uniform sampling distorts geography and creates unrealistic weather distributions.

#### Wheat
```python
WHEAT_DISTRICTS = {
    "Punjab":   0.40,   # Ludhiana, Amritsar, Patiala, Jalandhar
    "Haryana":  0.30,   # Karnal, Hisar, Rohtak, Ambala
    "UP":       0.20,   # Meerut, Agra, Bareilly, Lucknow
    "MP":       0.07,   # Hoshangabad, Vidisha, Sehore
    "Rajasthan":0.03,   # Sri Ganganagar, Hanumangarh
}
```

#### Rice
```python
RICE_DISTRICTS = {
    "WestBengal":   0.30,   # Burdwan, Nadia, Murshidabad
    "Odisha":       0.20,   # Cuttack, Puri, Sambalpur
    "AndhraPradesh":0.20,   # Krishna, Godavari, Guntur
    "TamilNadu":    0.15,   # Thanjavur, Tiruchirappalli
    "Punjab":       0.10,   # (kharif rice) Amritsar, Gurdaspur
    "Chhattisgarh": 0.05,   # Raipur, Durg
}
```

#### Sugarcane
```python
SUGARCANE_DISTRICTS = {
    "UP":           0.45,   # Meerut, Muzaffarnagar, Saharanpur, Bijnor
    "Maharashtra":  0.25,   # Pune, Kolhapur, Nashik, Solapur
    "Karnataka":    0.15,   # Belagavi, Bagalkot
    "TamilNadu":    0.15,   # Erode, Tirupur, Cuddalore
}
```

#### Cotton
```python
COTTON_DISTRICTS = {
    "Maharashtra":  0.40,   # Nagpur, Amravati, Yavatmal, Akola
    "Gujarat":      0.35,   # Rajkot, Surendranagar, Bhavnagar
    "Telangana":    0.25,   # Warangal, Adilabad, Karimnagar
}
```

---

## PART 4 — DISEASE-TO-ENVIRONMENT MAPPING (Probabilistic Windows)

> **Key design change from V1**: Disease-season windows are now **probabilistic**, not deterministic.
> Real diseases overlap seasons, vary by region, and shift with climate year-to-year.
> Use weighted random sampling from the window distribution for each disease.

### How to sample a window for a given disease:
```python
import random

def sample_window(disease_windows: dict) -> tuple:
    """
    disease_windows = {
        "Jan-Feb": 0.70,
        "Dec-Mar": 0.30,
    }
    Returns: (start_date, end_date) — a 30-day window within sampled period
    """
    period = random.choices(
        list(disease_windows.keys()),
        weights=list(disease_windows.values())
    )[0]
    # Parse period → random 30-day slice within it
    ...
```

---

### Wheat Disease Windows

| Disease | Primary Window (weight) | Secondary Window (weight) | Favorable Conditions |
|---|---|---|---|
| Yellow Rust | Jan–Feb (0.70) | Dec–Mar (0.30) | Temp 10–15°C, RH >80%, light rain |
| Brown Rust | Feb–Mar (0.65) | Jan–Apr (0.35) | Temp 15–22°C, RH >70% |
| Black Rust | Mar–Apr (0.75) | Feb–Apr (0.25) | Temp 20–25°C, RH >70% |
| Mildew | Dec–Feb (0.60) | Nov–Mar (0.40) | Temp 10–18°C, moderate RH, dry spells |
| Septoria | Feb–Mar (0.70) | Jan–Apr (0.30) | Temp 15–20°C, >20mm rain, wet periods |
| Tan Spot | Mar–Apr (0.65) | Feb–Apr (0.35) | Temp 20–28°C, wet+dry cycles |
| Fusarium Head Blight | Mar–Apr (0.80) | Feb–Apr (0.20) | Temp 25–30°C, RH >90%, rain at heading |
| Smut | Nov–Dec (0.70) | Oct–Jan (0.30) | Cool + moist at sowing |
| Aphid | Jan–Mar (0.60) | Dec–Apr (0.40) | Temp 15–25°C, low wind |
| Mite | Mar–Apr (0.70) | Feb–May (0.30) | Temp >25°C, RH <50%, dry |
| Blast | Feb–Mar (0.65) | Jan–Apr (0.35) | Temp 24–28°C, RH >89%, long leaf wetness |
| Common Root Rot | Nov–Jan (0.70) | Oct–Feb (0.30) | Soil temp >15°C, dry after wet |
| Stem Fly | Nov–Jan (0.65) | Oct–Feb (0.35) | Temp 15–25°C, moderate humidity |
| Leaf Blight | Mar–Apr (0.75) | Feb–May (0.25) | High temp + high humidity |
| Healthy | Uniform random across full season | — | — |

---

### Rice Disease Windows

| Disease | Primary Window (weight) | Secondary Window (weight) | Favorable Conditions |
|---|---|---|---|
| Leaf Blast | Aug–Sep (0.70) | Jul–Oct (0.30) | Temp 24–28°C, RH >89%, N-excess |
| Bacterial Leaf Blight | Jul–Aug (0.65) | Jun–Sep (0.35) | Temp 25–34°C, flooding + wind |
| Brown Spot | Sep–Oct (0.60) | Aug–Nov (0.40) | Temp 25–30°C, drought stress, low K |
| Sheath Blight | Aug–Sep (0.70) | Jul–Oct (0.30) | Temp 28–32°C, RH >95%, dense canopy |
| Leaf Scald | Sep–Oct (0.65) | Aug–Nov (0.35) | Temp 25–30°C, alternating wet/dry |
| Healthy Rice Leaf | Uniform random | — | — |

---

### Sugarcane Disease Windows

| Disease | Primary Window (weight) | Secondary Window (weight) | Favorable Conditions |
|---|---|---|---|
| Redrot | Jul–Sep (0.70) | Jun–Oct (0.30) | Temp 25–33°C, RH >80%, poor drainage |
| Mosaic | Jun–Aug (0.65) | May–Sep (0.35) | Temp 20–30°C, aphid vector activity |
| Rust | Sep–Nov (0.70) | Aug–Dec (0.30) | Temp 18–24°C, RH >80%, dew |
| Yellow Disease | Oct–Dec (0.70) | Sep–Dec (0.30) | Cool nights <18°C, leafhopper activity |
| Healthy | Uniform random | — | — |

---

### Cotton Disease Windows

| Disease | Primary Window (weight) | Secondary Window (weight) | Favorable Conditions |
|---|---|---|---|
| Bacterial Blight | Jul–Aug (0.70) | Jun–Sep (0.30) | Temp 25–35°C, high RH, rain splash |
| Curl Virus | Aug–Sep (0.65) | Jul–Oct (0.35) | Temp 25–35°C, whitefly vector, dry |
| Leaf Reddening | Sep–Oct (0.65) | Aug–Nov (0.35) | Temp >35°C, moisture stress |
| Leaf Variegation | Aug–Oct (0.60) | Jul–Nov (0.40) | Moderate temp, nutrient imbalance |
| Leaf Hopper Jassids | Jul–Sep (0.65) | Jun–Oct (0.35) | Temp 25–35°C, dry conditions |
| Herbicide Damage | Uniform random | — | Not weather-driven |
| Healthy Leaf | Uniform random | — | — |

---

## PART 5 — AGENT EXECUTION INSTRUCTIONS

### Full step-by-step pipeline

```
FOR each image in dataset:

  1. READ:
       crop_type, disease_label  ← from image metadata

  2. LOOK UP disease mapping (Part 4):
       window_distribution  ← dict of {period: probability}
       favorable_states     ← list from Part 3

  3. SAMPLE district (probabilistic, NOT uniform):
       state    ← weighted random from CROP_DISTRICTS distribution
       district ← random from districts list in that state
       lat, lon ← centroid from shapefile

  4. SAMPLE time window (probabilistic):
       period       ← weighted random from window_distribution
       start_date   ← random date within period such that start_date + 30 ≤ period end
       end_date     ← start_date + 30 days

  5. CALL Open-Meteo Historical API:
       URL: https://archive-api.open-meteo.com/v1/archive
       params:
         latitude=lat, longitude=lon
         start_date=start_date, end_date=end_date
         hourly=temperature_2m,relative_humidity_2m,dew_point_2m,
                precipitation,wind_speed_10m,soil_temperature_0cm
         timezone=Asia/Kolkata

  6. AGGREGATE hourly → daily (11 columns: 9 weather + 2 season encoding)

  7. STORE weather as numpy array shape [30, 11]
       filename: {sample_id}_weather.npy

  8. SAVE record to dataset_index.json:
       {
         "id":             "000001",
         "image_path":     "images/wheat/yellow_rust_001.jpg",
         "crop_type":      "wheat",
         "disease_label":  "Yellow Rust",
         "district":       "Ludhiana",
         "state":          "Punjab",
         "lat":            30.9010,
         "lon":            75.8573,
         "start_date":     "2022-01-15",
         "end_date":       "2022-02-14",
         "split":          "train"       ← assigned at district level, NOT image level
       }
```

---

### Filesystem layout

```
/dataset/
  dataset_index.json          ← master record list with split assignments
  weather/
    000001_weather.npy        ← shape [30, 11], float32
    000002_weather.npy
    ...
  images/
    wheat/
    rice/
    sugarcane/
    cotton/
  splits/
    wheat_district_splits.json    ← {district: "train"|"val"|"test"}
    rice_district_splits.json
    sugarcane_district_splits.json
    cotton_district_splits.json
```

---

### Weather normalization

Compute **z-score statistics on training set only**. Save stats. Apply same stats to val/test.

```python
# Compute on train split
train_wx = np.stack([np.load(f) for f in train_weather_files])  # [N, 30, 11]
mean = train_wx.mean(axis=(0, 1))   # [11]
std  = train_wx.std(axis=(0, 1))    # [11]
np.save("weather_stats.npy", {"mean": mean, "std": std})

# Apply everywhere
wx_normalized = (wx - mean) / (std + 1e-8)
```

> Do NOT compute normalization stats on the full dataset — that leaks val/test distribution into training.

---

## PART 6 — MODEL ARCHITECTURE

### Overview

Dual-stream late-fusion model. Two independent encoders, concatenated before classifier.

```
Image (224×224×3)                 Weather Sequence (30×11)
        |                                   |
  EfficientNetV2-S                  1D CNN Encoder
   (ImageNet pretrained)             Conv1d(64, k=3)
        |                             BatchNorm + ReLU
  GlobalAvgPool [B, 1280]            Conv1d(128, k=3)
        |                             BatchNorm + ReLU
  Linear(512) + ReLU                        |
  Dropout(0.3)                    Self-Attention (4 heads)
        |                            AdaptiveAvgPool1d(1)
      [B, 512]                        Linear(128) + ReLU
                                            |
                                         [B, 128]
                                            |
                    ┌───────────────────────┘
                    │      Concatenate → [B, 640]
                    │
             Linear(256) + ReLU
               Dropout(0.4)
             Linear(num_classes)
```

---

### Why 1D CNN + Attention (not LSTM, not MLP)

| Choice | Problem |
|---|---|
| MLP | Ignores temporal ordering — "5 rainy days in a row" looks same as "5 scattered rainy days." Disease epidemiology is pattern-sensitive. |
| LSTM/GRU | Overkill for 30 timesteps. Slow, harder to train, vanishing gradient still occurs. No evidence of better performance on short weather sequences. |
| **1D CNN + Attention** ✓ | CNN kernels (k=3, k=5) detect local weather bursts (3–7 day humidity streaks, rainfall events). Attention then learns which of those bursts mattered most for disease onset. Fast, interpretable, matches the biological mechanism. |

---

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import timm

class WeatherEncoder(nn.Module):
    def __init__(self, n_features=11, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 4-head attention over 30 timesteps
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True,
                                          dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, out_dim)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        # x: [B, 30, 11]
        x = x.permute(0, 2, 1)           # [B, 11, 30]
        x = self.conv(x)                  # [B, 128, 30]
        x = x.permute(0, 2, 1)           # [B, 30, 128]
        attn_out, _ = self.attn(x, x, x) # [B, 30, 128]
        x = self.norm(x + attn_out)      # residual + LayerNorm
        x = x.permute(0, 2, 1)           # [B, 128, 30]
        x = self.pool(x).squeeze(-1)      # [B, 128]
        return self.fc(x)                 # [B, out_dim]


class MultimodalDiseaseNet(nn.Module):
    def __init__(self, num_classes, weather_dim=128, img_dim=512):
        super().__init__()
        self.image_encoder = timm.create_model(
            'efficientnetv2_s', pretrained=True, num_classes=0
        )  # → [B, 1280]
        self.image_fc = nn.Sequential(
            nn.Linear(1280, img_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.weather_encoder = WeatherEncoder(n_features=11, out_dim=weather_dim)
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + weather_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, weather):
        img_feat = self.image_fc(self.image_encoder(image))   # [B, 512]
        wx_feat  = self.weather_encoder(weather)               # [B, 128]
        fused    = torch.cat([img_feat, wx_feat], dim=1)       # [B, 640]
        return self.classifier(fused)
```

---

## PART 7 — PYTORCH DATALOADER

```python
import os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

DISEASE_MAPS = {
    "wheat": {
        "Aphid": 0, "Black Rust": 1, "Blast": 2, "Brown Rust": 3,
        "Common Root Rot": 4, "Fusarium Head Blight": 5, "Healthy": 6,
        "Leaf Blight": 7, "Mildew": 8, "Mite": 9, "Septoria": 10,
        "Smut": 11, "Stem fly": 12, "Tan spot": 13, "Yellow Rust": 14
    },
    "rice": {
        "Bacterial Leaf Blight": 0, "Brown Spot": 1, "Healthy Rice Leaf": 2,
        "Leaf Blast": 3, "Leaf scald": 4, "Sheath Blight": 5
    },
    "sugarcane": {"Healthy": 0, "Mosaic": 1, "Redrot": 2, "Rust": 3, "Yellow": 4},
    "cotton": {
        "Bacterial blight": 0, "Curl virus": 1, "Healthy leaf": 2,
        "Herbicide growth damage": 3, "Leaf hopper jassids": 4,
        "Leaf redding": 5, "Leaf variegation": 6
    }
}

WEATHER_STATS_PATH = "dataset/weather_stats.npy"  # computed on train only

class CropDiseaseDataset(Dataset):
    def __init__(self, index_path, weather_dir, split="train", transform=None):
        with open(index_path) as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r["split"] == split]
        self.weather_dir = weather_dir
        self.transform   = transform or self._default_transform(split)
        stats = np.load(WEATHER_STATS_PATH, allow_pickle=True).item()
        self.wx_mean = torch.tensor(stats["mean"], dtype=torch.float32)  # [11]
        self.wx_std  = torch.tensor(stats["std"],  dtype=torch.float32)  # [11]

    def _default_transform(self, split):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        img = self.transform(img)
        wx = np.load(os.path.join(self.weather_dir, f"{r['id']}_weather.npy"))
        wx = torch.tensor(wx, dtype=torch.float32)          # [30, 11]
        wx = (wx - self.wx_mean) / (self.wx_std + 1e-8)    # normalize
        label = DISEASE_MAPS[r["crop_type"]][r["disease_label"]]
        return img, wx, label

    def get_class_weights(self):
        """For WeightedRandomSampler — handles class imbalance."""
        from collections import Counter
        labels = [DISEASE_MAPS[r["crop_type"]][r["disease_label"]] for r in self.records]
        counts = Counter(labels)
        n_total = len(labels)
        weights = [n_total / counts[l] for l in labels]
        return torch.tensor(weights, dtype=torch.float32)


# Usage with WeightedRandomSampler (recommended for imbalanced classes)
def build_dataloader(index_path, weather_dir, split, batch_size=32):
    ds = CropDiseaseDataset(index_path, weather_dir, split)
    if split == "train":
        sampler = WeightedRandomSampler(ds.get_class_weights(), len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)
```

---

## PART 8 — TRAIN / VAL / TEST SPLIT STRATEGY

**NEVER split randomly by image.** This causes district leakage (model learns `district → disease` shortcut).

### District-based split (per crop)

```
Train  → 70% of districts
Val    → 15% of districts
Test   → 15% of districts  ← completely held out, never seen during training
```

Assign splits at the **district level**, then propagate to all records from that district.

```python
import random

def assign_district_splits(district_list, seed=42):
    random.seed(seed)
    random.shuffle(district_list)
    n = len(district_list)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)
    return {
        d: "train" for d in district_list[:train_end]
    } | {
        d: "val"   for d in district_list[train_end:val_end]
    } | {
        d: "test"  for d in district_list[val_end:]
    }
```

Save each crop's district splits to `splits/{crop}_district_splits.json` before generating any records.

---

## PART 8B — OVERFITTING-AWARE DATA COLLECTION SAFEGUARDS

> **Why this section exists**: District-based splitting (Part 8) prevents *geographic* leakage, but it does
> not address several other, more subtle ways this specific pipeline can let the model memorize shortcuts
> instead of learning real disease signal. Each subsection below names a concrete failure mode created by
> the data collection design in Parts 1–7, and the specific control to apply **during collection**, not
> only during training. Apply these checks before generating the final `dataset_index.json`.

### 8B.1 — Image-side leakage: background and capture-condition shortcuts

**Failure mode**: Public leaf-disease image sources (PlantVillage-style) are often captured in batches —
one shooting session, one background, one lighting rig per disease/date. A CNN can learn to classify from
the background color, lighting, or camera artifacts alone, with zero attention to the actual lesion. This
is a documented, measured problem: a probe classifier trained on **8 background pixels only** from the
PlantVillage dataset reached **49.0% accuracy** on held-out data against a **2.6% random-chance baseline**, demonstrating the dataset contains background noise correlated with the labels that a model can exploit instead of learning real lesion features [Noyan, 2022 — see references].

**Controls to apply at collection time**:
- **Segment or mask backgrounds** where source metadata allows it, or record a `background_id` / `capture_batch_id` field per image so batches can be checked against labels for correlation before training.
- **Run the background-probe test yourself** before trusting the dataset: crop out an 8×8 corner patch from each image (leaf-free region), train a trivial linear classifier on just those pixels, and report its accuracy against chance. If it is well above chance, either segment leaves out or drop/reweight the offending source images.
- **Never let `capture_batch_id` (or its proxy, image filename prefix / upload timestamp) align with `split`** — if a batch is entirely disease X and entirely in one split, that split assignment itself becomes a leak vector independent of geography.
- **Log EXIF/capture metadata (camera model, timestamp, resolution) per image** and check its distribution per class; skewed metadata-to-label correlation is a leading indicator of shortcut risk even before training.

### 8B.2 — Weather-side leakage: the synthetic-conditioning shortcut

**Failure mode specific to this pipeline**: because weather is *assigned by disease-favorable priors*
(Part 4), a disease's window distribution is itself a partial encoding of its own label — the model can
learn "date range / temperature band → label" from the weather branch without the weather branch ever
representing anything physically true about that image. This is exactly the shortcut-learning risk the
methodology in Part 11 asks you to disclose, and it needs an explicit collection-time control, not just a
framing caveat in the paper.

- **Overlap windows across diseases within a crop deliberately.** Where two diseases' agronomic priors
  legitimately overlap (e.g., Wheat Mildew Dec–Feb vs. Yellow Rust Jan–Feb), do not artificially separate
  their sampled windows to make classes "cleaner" — that manufactures separability that has nothing to do
  with the image. Sample strictly from the probability tables in Part 4 as given.
- **Add jitter noise to the weather sequence during training** (not just image augmentation) so the model
  cannot memorize exact synthetic value patterns tied to a window. Time-series jittering (adding small
  Gaussian noise to each timestep) is a standard, empirically studied regularizer for sensor-style sequence
  data — jittering mimics high-frequency sensor noise, forcing the model to rely on shape rather than exact
  values [Um et al., 2017 — see references]:
  ```python
  def jitter(wx, sigma=0.03):
      # wx: [30, 11] already z-score normalized
      return wx + torch.randn_like(wx) * sigma
  ```
  Apply only at train time, after normalization, with a small sigma (start at 0.02–0.05 of the normalized
  scale) — large enough to blur exact synthetic signatures, small enough not to erase real seasonal shape.
- **Run a weather-ablation permutation check as a mandatory pre-training diagnostic**: shuffle the weather
  tensor across samples (break the image↔weather pairing) while keeping images and labels intact, then
  train the weather-only branch alone. If weather-only accuracy on shuffled pairings still exceeds a random
  baseline by a wide margin, the weather branch is encoding label information through the window-sampling
  procedure itself rather than through anything image-correlated — revisit the window overlap settings in
  Part 4 before proceeding.
- **Cap the number of samples drawn from any single (district, 30-day-window-start) pair.** Without a cap,
  probabilistic sampling (Part 3/4) can repeatedly draw near-identical weather sequences for the same
  disease, which is functionally close to duplicating training examples. Recommended cap: no more than
  ~1–2% of a class's total samples from any single (district, window-start) combination; resample if the
  cap is hit.

### 8B.3 — Near-duplicate image detection

**Failure mode**: multiple images of the same physical leaf (different angles/crops) ending up split
across train and val/test inflates validation performance without real generalization — a known issue in
leaf-disease datasets, where the same physical leaf is often photographed from multiple orientations; current
maintained releases of PlantVillage now ship pre-defined splits that explicitly respect leaf-grouping to
prevent this exact leakage [Mohanty et al., 2016; PlantVillage-Dataset repository — see references].

- **Compute a perceptual hash (e.g. pHash/dHash) per image** at collection time and store it in
  `dataset_index.json`. Flag any pair with Hamming distance below a small threshold (start at 5) as
  candidate duplicates.
- **Group flagged near-duplicates by a shared `leaf_group_id`** and assign the whole group to one split —
  exactly the same principle already used for district-level split assignment in Part 8, just applied at
  the leaf/image level in addition to the geographic level.

### 8B.4 — Spatial and temporal validation beyond a single 70/15/15 split

**Failure mode**: a single random district partition can, by chance, produce a train/test split that is
easier or harder than typical, especially with weather features that carry real spatial autocorrelation
(nearby districts have correlated weather). Reporting one split's numbers overstates confidence. This is
the general problem block cross-validation was designed to address for spatially/temporally structured
data: block cross-validation, where data are split strategically rather than randomly, addresses spatial
dependence issues that persist even when the model itself already accounts for autocorrelation
[Roberts et al., 2017 — see references].

- **Repeat the district-level split (Part 8) with ≥5 different seeds** and report mean ± std of macro-F1
  across seeds, not a single number.
- **Add one genuinely held-out spatial block**: reserve an entire *state* not used in Parts 3's probability
  tables (e.g. a minor wheat-growing state excluded from `WHEAT_DISTRICTS`) purely as an external test set,
  never touched during model or hyperparameter selection.
- **Add a temporal holdout in addition to the spatial one**: if your image source spans multiple years,
  hold out one full crop year (e.g. train on 2018–2021 seasons, test on 2022) so the reported metric also
  reflects generalization to a weather year the model never trained on — not just a district it never
  trained on. Note this explicitly as a separate ablation from the district-based split.

### 8B.5 — Mandatory sanity check before reporting any result

Run a **label-randomization control** on a small subset (e.g. 10% of train) before trusting any headline
number: shuffle the disease labels so they are decoupled from both image and weather, then confirm
training accuracy collapses toward chance and does not silently reach high values. This test exists because
deep networks are known to be able to fit arbitrary label noise given enough capacity, which means high
train accuracy alone is not evidence the model has learned anything about disease–environment structure:
standard CNNs can reach 0% training error on completely randomly-labeled data while test error stays at
chance level, precisely because there is no real correlation between training and test labels to learn
[Zhang et al., 2017 — see references]. If shuffled-label training accuracy climbs anywhere close to
real-label training accuracy, your effective model capacity relative to dataset size is too high for the
current data volume — reduce capacity, increase regularization, or grow the dataset before proceeding to
full training.

---

## PART 9 — TRAINING CONFIGURATION

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Loss ---
# Option A (default): label smoothing handles weak label noise
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Option B (if class imbalance is severe after V1 — especially sugarcane/cotton):
# from torchvision.ops import sigmoid_focal_loss  # or implement manually
# Use focal loss with gamma=2, alpha per-class

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# --- Training loop structure ---
EPOCHS     = 50
BATCH_SIZE = 32
WARMUP     = 5  # epochs with frozen image encoder

# Phase 1: Warm up weather branch (freeze image encoder)
for epoch in range(WARMUP):
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    # train loop ...

# Phase 2: Full fine-tuning (unfreeze all)
for param in model.image_encoder.parameters():
    param.requires_grad = True
# resume training loop ...
```

> **Why freeze first**: The image encoder is already pretrained on ImageNet. If both branches train simultaneously from epoch 1, the image branch dominates early gradients and the weather branch never gets meaningful training signal. The 5-epoch freeze lets the weather encoder reach a useful embedding space before joint training.

---

## PART 10 — EVALUATION METRICS

Report all of the following per crop:

| Metric | Tool | Why |
|---|---|---|
| Per-class F1 | `sklearn.metrics.classification_report` | Shows which diseases the model struggles with |
| Macro F1 | Same | Treats all diseases equally — critical for rare diseases |
| Weighted F1 | Same | Overall performance weighted by class frequency |
| Confusion Matrix | `sklearn.metrics.confusion_matrix` | Reveals rust-type confusion (Yellow/Brown/Black look similar) |
| AUC-ROC (per class) | `sklearn.metrics.roc_auc_score` | Threshold-independent, useful for imbalanced classes |

### Required ablation study (non-negotiable for paper)

Run **3 model variants** and compare:

| Variant | Description |
|---|---|
| Image-only | EfficientNetV2-S alone, no weather input |
| Weather-only | 1D CNN + Attention alone, no image |
| **Multimodal (full)** | Combined — this is your main model |

This ablation proves whether the environmental branch actually contributes. Without it, reviewers will reject the paper's core claim.

---

## PART 11 — PAPER FRAMING & METHODOLOGY SECTION OUTLINE

### ⚠️ How to frame your contribution

**DO NOT say**: "We built a real-time environmental disease prediction system"
**DO say**: "We propose a context-aware disease classification framework using weakly aligned environmental priors derived from historical reanalysis data"

This framing is both accurate and scientifically stronger.

---

### Methods section structure (follow this order)

**3.1 Dataset and Multimodal Data Acquisition**
- PlantVillage images as visual source (cite original PlantVillage paper)
- Open-Meteo Historical Archive API as environmental source (cite)
- District shapefile source (ICRISAT or Survey of India)
- Acknowledge: images and weather are NOT co-located — this is weakly aligned environmental conditioning

**3.2 Crop Calendar and Spatial Mapping**
- Table: crop → districts → growth phases → disease windows
- District probability distributions (Table X)
- Centroid extraction methodology

**3.3 Disease-to-Environment Prior Mapping**
- Agronomic justification for each disease's favorable conditions (cite ICAR guidelines, FAO crop protection manuals)
- Probabilistic window sampling design (explain weighted random sampling)
- Explicitly name the approach: "synthetic environmental association"

**3.4 Data Preprocessing**
- Image: resize to 224×224, augmentation (random crop, horizontal flip, color jitter, Gaussian blur), ImageNet normalization
- Weather: hourly → daily aggregation, z-score normalization (train statistics only), season encoding (month_sin, month_cos)
- Weather shape: [30 days × 11 features]

**3.5 Model Architecture**
- Dual-stream: EfficientNetV2-S (image) + 1D CNN with Self-Attention (weather)
- Late fusion via concatenation
- Diagram of architecture

**3.6 Training Protocol**
- District-based 70/15/15 split
- Phase 1: image encoder frozen (5 epochs), Phase 2: full fine-tuning (45 epochs)
- AdamW + cosine annealing, label smoothing 0.1
- WeightedRandomSampler for class balance

**3.7 Generalization Safeguards** *(new — cite Part 8B)*
- Background-shortcut probe (8-pixel corner classifier) and its result
- Weather-shuffle permutation test result confirming the environmental branch isn't just decoding the
  window-sampling procedure
- Near-duplicate leaf grouping via perceptual hashing
- Multi-seed district split (mean ± std) plus one fully held-out state and one held-out crop year
- Label-randomization sanity check result

**3.8 Evaluation**
- Metrics: per-class F1, macro F1, confusion matrix, AUC-ROC
- Ablation: image-only vs weather-only vs multimodal

---

## QUICK REFERENCE — FINAL DECISIONS TABLE

| Decision | Choice | Reason |
|---|---|---|
| Weather API | Open-Meteo Historical Archive | Free, reliable, pan-India coverage |
| API resolution | Hourly → daily aggregation | Daily sufficient, reduces noise |
| V1 features | 9 weather + 2 season encoding = 11 total | Core reliable variables only |
| Sequence length | 30 days | Captures incubation + spread; not too noisy |
| District sampling | Probabilistic by crop dominance | Reflects real agricultural geography |
| Disease windows | Probabilistic (weighted) | Reflects real seasonal overlap |
| Environmental model | 1D CNN + Self-Attention | Local burst detection + temporal weighting |
| Image model | EfficientNetV2-S pretrained | Best accuracy/speed; good for 224×224 |
| Fusion | Late concatenation | Simple, interpretable, standard in literature |
| Train/test split | District-based 70/15/15 | Prevents geographic leakage |
| Class imbalance | WeightedRandomSampler + label smoothing | Handles disease frequency skew |
| Loss | CrossEntropy (label_smoothing=0.1) | Handles weak label noise |
| Paper framing | "Weakly aligned environmental conditioning" | Accurate AND defensible to reviewers |
| V1 → V2 upgrade | Add VPD, soil moisture only after V1 works | Validate complexity incrementally |
| Background leakage check | 8×8-pixel background probe before training | Detects PlantVillage-style shortcut learning (Part 8B.1) |
| Weather leakage check | Shuffled-pairing permutation test + jitter augmentation | Detects synthetic window→label shortcut (Part 8B.2) |
| Duplicate leaves | pHash/dHash grouping into one split | Prevents near-duplicate train/test overlap (Part 8B.3) |
| Split robustness | ≥5-seed district split + held-out state + held-out year | Single split understates variance (Part 8B.4) |
| Sanity check | Label-randomization control before trusting results | Confirms model isn't just memorizing (Part 8B.5) |

---

## REFERENCES — DISEASE FAVORABLE CONDITIONS

These are the sources used to establish the environmental thresholds (temperature, humidity, season windows) in Part 4. Cite these in your paper's Methods section when justifying the disease-to-environment mapping.

---

### WHEAT

#### Yellow Rust (Stripe Rust) — *Puccinia striiformis* f. sp. *tritici*
- **Wikipedia — Wheat Yellow Rust**
  Temp 2–15°C (max 23°C), high humidity and rainfall favor infection.
  https://en.wikipedia.org/wiki/Wheat_yellow_rust

- **AHDB (Agriculture and Horticulture Development Board) — Yellow Rust**
  10–15°C with RH ~100% optimal for spore germination and spread. Temperatures >20°C slow the fungus.
  https://ahdb.org.uk/yellowrust

- **Crop Protection Network — An Overview of Stripe Rust of Wheat**
  Optimal 10–18°C (50–64°F), intermittent rain or dew events.
  https://cropprotectionnetwork.org/publications/an-overview-of-stripe-rust-of-wheat

- **American Phytopathological Society (Phytopathology) — Disease–Weather Relationships for Powdery Mildew and Yellow Rust on Winter Wheat**
  Temperature in Feb–Jun was the most influential factor for a damaging epidemic of yellow rust.
  https://apsjournals.apsnet.org/doi/10.1094/PHYTO-98-5-0609 *(PubMed: PMID 18943230)*

#### Brown Rust (Leaf Rust) — *Puccinia triticina*
- **Ohio State University Extension (Ohioline) — Rust Diseases of Wheat**
  Temp 77–86°F (25–30°C), heavy dew, high humidity. Leaf rust develops at warmer temps than stripe rust.
  https://ohioline.osu.edu/factsheet/plpath-cer-12

- **Bayer Crop Science — Wheat Rust Diseases**
  59–68°F (15–20°C) optimal; min 6 hours moisture on leaf surface.
  https://www.cropscience.bayer.us/articles/cp/wheat-rust-diseases

#### Black Rust (Stem Rust) — *Puccinia graminis* f. sp. *tritici*
- **Bayer Crop Science — Wheat Rust Diseases** *(same source as above)*
  Warmest conditions of the three rusts: 59–84°F (15–29°C), 6–8 hours moisture. Hot days (77–86°F) and high nighttime humidity.

- **PMC — Exploring the Frontier of Wheat Rust Resistance**
  Stem rust: 25–30°C day, 15–20°C night, elevated relative humidity.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11396821/

- **FAO — Stem Rust (Will Stem Rust Destroy the World's Wheat Crop?)**
  6–8 hours dew or free moisture from rain required for spore germination.
  https://www.fao.org/fileadmin/user_upload/rust/docs/Singh%20et%20al%20Advances%20in%20Agronomy%20Vol_1.%2098.pdf

#### Powdery Mildew — *Blumeria graminis* f. sp. *tritici*
- **FAO — Compendium of Wheat Diseases and Pests (Cunfer chapter)**
  Cool humid conditions; important where rain occurs early and temperatures are relatively cool; maritime and humid continental climates.
  https://www.fao.org/4/y4011e/y4011e0l.htm

#### Fusarium Head Blight (Scab) — *Fusarium graminearum*
- **Crop Protection Network — An Overview of Fusarium Head Blight**
  Warm, wet, humid weather for 48–72 hours during flowering. Primary risk window is at anthesis.
  https://cropprotectionnetwork.org/publications/an-overview-of-fusarium-head-blight

- **Purdue University Extension — Foliar Diseases of Wheat and FHB Management**
  Warm, wet weather with high RH at anthesis. Risk highest with 3+ days of RH >70% and 65–80°F.
  https://extension.entm.purdue.edu/newsletters/pestandcrop/article/foliar-diseases-of-wheat-and-fusarium-head-blight-scab-management-5/

- **Bayer Crop Science — Fusarium Head Blight in Wheat**
  RH >70% for 36+ hours, temperatures 68–77°F (20–25°C) during flowering.
  https://www.cropscience.bayer.us/articles/bayer/fusarium-head-blight-in-wheat

- **PMC — Fusarium Head Blight on Wheat: Biology, Detection, and Integrated Management**
  Rainy days + warm temps at anthesis. Temp range 16–30°C, optimum 25–28°C for *F. graminearum*.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10053988/

- **Government of Saskatchewan — Fusarium Head Blight**
  Precipitation or high humidity for ≥12 hours required for germination. Optimum 25–28°C.
  https://www.saskatchewan.ca/business/agriculture-natural-resources-and-industry/agribusiness-farmers-and-ranchers/crops-and-irrigation/disease/fusarium-head-blight

#### Septoria Leaf Blotch — *Zymoseptoria tritici*
- **Crop Protection Network — Septoria Tritici Blotch of Wheat** *(referenced within Purdue newsletter above)*
  Cool, wet conditions. Temp 15–20°C, prolonged wet periods (>20mm rain).
  https://cropprotectionnetwork.org/encyclopedia/septoria-tritici-blotch-of-wheat

#### Tan Spot — *Pyrenophora tritici-repentis*
- **Crop Protection Network — Tan Spot of Wheat** *(referenced within Purdue newsletter above)*
  Warm (20–28°C), wet then dry cycles. Spores released during wet periods.
  https://cropprotectionnetwork.org/encyclopedia/tan-spot-of-wheat

---

### RICE

#### Leaf Blast — *Magnaporthe oryzae*
- **UC IPM — Rice Blast (UC Statewide IPM Program)**
  RH >93% for sporulation; optimum 25–26°C (77–82°F). Long periods of free moisture; high humidity, little wind at night.
  https://ipm.ucanr.edu/agriculture/rice/rice-blast/

- **PMC — Defending Rice Crop from Blast Disease (Nepal)**
  Optimal conditions: high humidity, leaf wetness, temperatures 17–28°C.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12238087/

#### Bacterial Leaf Blight (BLB) — *Xanthomonas oryzae* pv. *oryzae*
- **IRRI Rice Knowledge Bank — Bacterial Blight**
  Temp 25–34°C, RH >70%. Strong winds and continuous heavy rains spread bacteria through ooze droplets.
  http://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight

- **PlantVillage (Penn State) — Rice Bacterial Blight**
  Same thresholds confirmed: 25–34°C, RH >70%.
  https://plantvillage.psu.edu/diseases/bacterial-blight-2c3e7c50-6320-4bf7-802a-27dceb60e63d

- **Lucid Apps Pacific Pests & Pathogens — Rice Bacterial Leaf Blight**
  25–34°C and RH >70%. Spread in irrigation, floodwater, and wind-driven rain.
  https://apps.lucidcentral.org/ppp/text/web_full/entities/rice_bacterial_leaf_blight_418.htm

#### Sheath Blight — *Rhizoctonia solani*
- **IRRI Rice Knowledge Bank — Sheath Blight**
  High temperature 28–32°C, RH 85–100%. Plants most vulnerable during rainy season.
  http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/sheath-blight

- **Wikipedia — Sheath Blight of Rice**
  High humidity and low sunlight favor lesion spread.
  https://en.wikipedia.org/wiki/Sheath_blight_of_rice

#### Brown Spot — *Cochliobolus miyabeanus*
  Drought stress, low potassium, temp 25–30°C. Broadly documented in IRRI literature (see Sheath Blight source above for IRRI general reference).

#### Leaf Scald — *Microdochium oryzae*
  Alternating wet/dry conditions, 25–30°C. See LSU AgCenter sugarcane/rice disease compendium:
  https://www.lsuagcenter.com/portals/communications/publications/management_guides/plant_disease_guide/field-crops-sugarcane

---

### SUGARCANE

#### Red Rot — *Colletotrichum falcatum* (syn. *Glomerella tucumanensis*)
- **Organica Biotech — Red Rot Disease in Sugarcane**
  High humidity (monsoon), warm temperatures 20–30°C.
  https://organicabiotech.com/red-rot-disease-in-sugarcane-and-its-management/

- **Katyayani Krishi Direct — Red Rot Disease in Sugarcane**
  Optimal temp 28–32°C, high relative humidity, waterlogged conditions.
  https://katyayanikrishidirect.com/blogs/news/measures-to-control-red-rot-in-sugarcane-crop

- **Lucid Apps Pacific Pests & Pathogens — Sugarcane Red Rot**
  Cool, wet weather and heavy soils; drought also increases susceptibility.
  https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/sugarcane_red_rot_221.htm

- **PMC — Combining Genomic Selection for Red Rot Resistance in Sugarcane**
  India context; large-scale epidemics since 1901. Confirms monsoon-season risk.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9660812/

#### Rust (Common Rust) — *Puccinia melanocephala*
- **Plantix — Sugarcane Common Rust**
  RH ~98%, cool nights followed by warm days 20–25°C. Continuous leaf wetness ≥9 hours.
  https://plantix.net/en/library/plant-diseases/100153/sugarcane-common-rust/

#### Mosaic — Sugarcane Mosaic Virus (SCMV), aphid-vectored
- **LSU AgCenter — Field Crops: Sugarcane (Disease Management)**
  Managed primarily with host resistance. Spread through planting infected material and aphid vectors.
  https://www.lsuagcenter.com/portals/communications/publications/management_guides/plant_disease_guide/field-crops-sugarcane

- **PMC — Sugarcane Pokkah Boeng Disease: Insights and Future Directions**
  Mosaic virus causes 40% yield loss globally. Aphid vector activity peaks in warm, humid periods.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11677414/

#### Yellow Disease — *Candidatus Phytoplasma sacchari*, leafhopper-vectored
  Cool nights <18°C promote leafhopper activity and transmission. See general ICAR sugarcane disease documentation. BigHaat and Katyayani sources above confirm seasonal pattern.

---

### COTTON

#### Bacterial Blight — *Xanthomonas citri* pv. *malvacearum*
- **Cotton Incorporated — Management of Bacterial Blight of Cotton**
  Daytime 90–100°F (32–38°C), nighttime 62–68°F. Heavy rainfall followed by warm humid conditions (RH >85%) greatly increases spread.
  https://www.cottoninc.com/cotton-production/ag-research/plant-pathology/management-bacterial-blight-cotton/

- **Bayer Crop Science — Bacterial Blight in Cotton**
  RH >85%, daytime 86–100°F (30–38°C), nighttime 62–68°F.
  https://www.cropscience.bayer.us/articles/dad/bacterial-blight-in-cotton

- **NC State Extension — Bacterial Blight of Cotton**
  Significant rainfall + high humidity + warm temperatures. Disease most severe after storms with heavy rain/hail.
  https://content.ces.ncsu.edu/bacterial-blight-angular-leaf-spot-of-cotton

- **LSU AgCenter — Bacterial Blight Observed in Louisiana Cotton Fields**
  RH >85%, temp 86–97°F (30–36°C) optimal.
  https://www.lsuagcenter.com/profiles/truffin/articles/page1567626832196

#### Cotton Leaf Curl Virus (CLCuV) — *Begomovirus*, whitefly (*Bemisia tabaci*) vectored
- **PMC — Temporal Changes in Virus/Betasatellite DNA in B. tabaci Feeding on CLCuD Affected Cotton**
  Whitefly vector. Temperature, humidity, and rainfall affect CLCuD. Hot and dry conditions favor whitefly proliferation.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11150673/

- **Taylor & Francis — New Fangled Tactics Towards Cotton Leaf Curl Virus Disease (Review)**
  Tropical areas with high humidity and elevated temperatures constrain management. Confirms temperature-vector interaction.
  https://www.tandfonline.com/doi/full/10.1080/15440478.2023.2217364

- **ScienceDirect — Cotton Leaf Curl Virus (overview)**
  Temperature, humidity, and rainfall are key environmental cues alongside whitefly vector presence.
  https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/cotton-leaf-curl-virus

#### Leaf Hopper Jassids — *Amrasca biguttula biguttula*, insect pest
  Warm (25–35°C), dry conditions favor population buildup. ICAR and state agriculture extension manuals document this consistently. See TNAU Agritech Portal for Indian context:
  http://eagri.org/eagri50/PATH272/lecture13/008.html *(covers cotton Bacterial Blight and pest interaction under Indian conditions)*

---

### GENERAL / CROSS-CROP REFERENCES

- **IRRI (International Rice Research Institute) — Rice Knowledge Bank**
  Comprehensive disease factsheet database for all major rice diseases.
  http://www.knowledgebank.irri.org

- **FAO — Compendium of Wheat Diseases and Pests (2nd Edition)**
  Covers wheat mildew, rusts, blights. Disease-weather interactions systematically documented.
  https://www.fao.org/4/y4011e/y4011e00.htm

- **Crop Protection Network (CPN)**
  Peer-reviewed extension publications covering FHB, stripe rust, tan spot, septoria.
  https://cropprotectionnetwork.org

- **PlantVillage (Penn State University)**
  Open-access disease fact sheets for all major crops including rice, wheat, cotton.
  https://plantvillage.psu.edu

- **Plantix (PEAT GmbH)**
  Disease library with environmental condition summaries for sugarcane rust and other crops.
  https://plantix.net/en/library/

> **Citation guidance for paper**: When citing disease favorable conditions in your Methods section, prefer IRRI, FAO, APS (American Phytopathological Society), and peer-reviewed PMC/PubMed sources over extension bulletins. Use extension sources (Purdue, AHDB, NC State) as supporting citations or for the specific numerical thresholds they document.

---

## REFERENCES — DATA COLLECTION METHODOLOGY & OVERFITTING MITIGATION (Part 8B)

These sources support the collection-time safeguards in Part 8B. Cite them in Methods §3.7 (or wherever
you justify the split strategy and leakage checks) alongside the disease-condition references above.

- **Noyan, M. A. (2022). "Uncovering Bias in the PlantVillage Dataset."** *arXiv:2206.04374.*
  A model trained on only 8 background pixels reached 49.0% accuracy vs. a 2.6% random baseline on
  PlantVillage, demonstrating label-correlated background bias. Direct justification for the 8B.1
  background-probe check.
  https://arxiv.org/pdf/2206.04374

- **Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). "Using Deep Learning for Image-Based Plant
  Disease Detection."** *Frontiers in Plant Science, 7:1419.*
  Original PlantVillage paper; explicitly notes multiple images per physical leaf and tests
  color/grayscale/segmented variants to probe for background-driven bias. Justifies 8B.1 and 8B.3.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5032846/

- **PlantVillage-Dataset maintained repository (spMohanty/PlantVillage-Dataset), GitHub.**
  Current releases provide predefined splits that "strictly respect the leaf grouping logic to prevent
  data leakage" — the precedent for the `leaf_group_id` deduplication control in 8B.3.
  https://github.com/spmohanty/plantvillage-dataset

- **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017).
  "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic
  Structure."** *Ecography, 40(8), 913–929.*
  Foundational reference for block/spatial cross-validation when data has geographic or temporal
  dependence. Justifies the multi-seed, held-out-state, and held-out-year controls in 8B.4.
  https://doi.org/10.1111/ecog.02881

- **Um, T. T., Pfister, F. M. J., Pichler, D., Endo, S., Lang, M., Hirche, S., Fietzek, U., & Kulić, D.
  (2017). "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using
  Convolutional Neural Networks."** *Proceedings of the 19th ACM International Conference on
  Multimodal Interaction (ICMI 2017), 216–220.*
  Establishes jittering (additive Gaussian noise per timestep) as a standard regularizer for
  short multivariate sensor sequences — the basis for the weather-jitter augmentation in 8B.2. Note
  for the paper: this same study also found jittering alone did not always improve accuracy in their
  task, so report your own ablation rather than assuming a gain.
  https://arxiv.org/abs/1706.00527

- **Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). "Understanding Deep Learning
  Requires Rethinking Generalization."** *International Conference on Learning Representations
  (ICLR 2017).*
  Shows standard CNNs can fit fully randomized labels to zero training error, which is why train
  accuracy alone cannot certify learning. Direct justification for the label-randomization sanity
  check in 8B.5.
  https://arxiv.org/abs/1611.03530
