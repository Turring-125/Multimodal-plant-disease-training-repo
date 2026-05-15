import os

base_path = r"d:\clg\major project\Project folder"
datasets = {
    "rice": "Rice_Leaf_AUG",
    "sugarcane": "Sugarcane Leaf Disease Dataset",
    "cotton": "cotton dataset",
    "wheat": "wheat"
}

for crop, folder in datasets.items():
    full_path = os.path.join(base_path, folder)
    if os.path.exists(full_path):
        print(f"\nCrop: {crop} ({folder})")
        subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
        print(f"Subdirs: {subdirs}")
    else:
        print(f"\nPath not found: {full_path}")
