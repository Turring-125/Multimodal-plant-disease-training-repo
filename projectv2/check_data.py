import json
from collections import Counter

data = json.load(open('dataset/dataset_index.json'))
splits = {'train': [], 'val': [], 'test': []}
for r in data:
    splits[r['split']].append(f"{r['crop_type']}_{r['disease_label']}")

print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

tc = Counter(splits['train'])
vc = Counter(splits['val'])

print("\n--- TRAIN class distribution ---")
for k, v in sorted(tc.items()):
    print(f"  {k}: {v}")

print("\n--- VAL class distribution ---")
for k, v in sorted(vc.items()):
    print(f"  {k}: {v}")

# Check total unified classes
CLASS_LIST = []
DISEASE_MAPS = {
    "wheat": {
        "Aphid": 0, "Black Rust": 1, "Blast": 2, "Brown Rust": 3,
        "Common Root Rot": 4, "Fusarium Head Blight": 5, "Healthy": 6,
        "Leaf Blight": 7, "Mildew": 8, "Mite": 9, "Septoria": 10,
        "Smut": 11, "Stem Fly": 12, "Tan Spot": 13, "Yellow Rust": 14
    },
    "rice": {
        "Bacterial Leaf Blight": 0, "Brown Spot": 1, "Healthy Rice Leaf": 2,
        "Leaf Blast": 3, "Leaf Scald": 4, "Sheath Blight": 5
    },
    "sugarcane": {"Healthy": 0, "Mosaic": 1, "Redrot": 2, "Rust": 3, "Yellow Disease": 4},
    "cotton": {
        "Bacterial Blight": 0, "Curl Virus": 1, "Healthy Leaf": 2,
        "Herbicide Damage": 3, "Leaf Hopper Jassids": 4,
        "Leaf Reddening": 5, "Leaf Variegation": 6
    }
}

for crop, m in DISEASE_MAPS.items():
    for disease in m.keys():
        CLASS_LIST.append(f"{crop}_{disease}")

print(f"\nTotal unified classes: {len(CLASS_LIST)}")
print("\nClasses in LABEL_MAP but NOT in train:")
for c in CLASS_LIST:
    if c not in tc:
        print(f"  MISSING from train: {c}")
    if c not in vc:
        print(f"  MISSING from val: {c}")
