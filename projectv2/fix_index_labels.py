import json
import os

# Copying the mapping logic from data_collection.py
DISEASE_KEYS = {
    "wheat": ["Yellow Rust", "Brown Rust", "Black Rust", "Mildew", "Septoria", "Tan Spot", "Fusarium Head Blight", "Smut", "Aphid", "Mite", "Blast", "Common Root Rot", "Stem Fly", "Leaf Blight", "Healthy"],
    "rice": ["Leaf Blast", "Bacterial Leaf Blight", "Brown Spot", "Sheath Blight", "Leaf Scald", "Healthy Rice Leaf"],
    "sugarcane": ["Redrot", "Mosaic", "Rust", "Yellow Disease", "Healthy"],
    "cotton": ["Bacterial Blight", "Curl Virus", "Leaf Reddening", "Leaf Variegation", "Leaf Hopper Jassids", "Herbicide Damage", "Healthy Leaf"]
}

def fix_index(index_path):
    with open(index_path, 'r') as f:
        records = json.load(f)
    
    fixed_count = 0
    for r in records:
        path = r['image_path']
        crop = r['crop_type']
        # Folder name is the parent of the image
        disease = os.path.basename(os.path.dirname(path))
        disease_norm = disease.strip().lower()
        
        found_label = None
        for l in DISEASE_KEYS[crop]:
            if l.lower() == disease_norm:
                found_label = l
                break
        
        # Manual overrides/fixes
        if not found_label:
            if crop == "sugarcane" and "red" in disease_norm: found_label = "Redrot"
            if crop == "sugarcane" and "yellow" in disease_norm: found_label = "Yellow Disease"
            if crop == "cotton" and "red" in disease_norm: found_label = "Leaf Reddening"
            if crop == "cotton" and "blight" in disease_norm: found_label = "Bacterial Blight"
            if crop == "cotton" and "herbicide" in disease_norm: found_label = "Herbicide Damage"
            if crop == "cotton" and "curl" in disease_norm: found_label = "Curl Virus"
            if crop == "cotton" and "hopper" in disease_norm: found_label = "Leaf Hopper Jassids"
            if crop == "cotton" and "healthy" in disease_norm: found_label = "Healthy Leaf"
            if crop == "cotton" and "variegation" in disease_norm: found_label = "Leaf Variegation"
            if crop == "rice" and "scald" in disease_norm: found_label = "Leaf Scald"

        if found_label and found_label != r['disease_label']:
            r['disease_label'] = found_label
            fixed_count += 1
            
    with open(index_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Fixed {fixed_count} records.")

if __name__ == "__main__":
    fix_index("dataset/dataset_index.json")
