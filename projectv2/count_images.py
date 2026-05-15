import os

base_path = r"d:\clg\major project\Project folder"
datasets = {
    "rice": os.path.join(base_path, "Rice_Leaf_AUG"),
    "sugarcane": os.path.join(base_path, "Sugarcane Leaf Disease Dataset"),
    "cotton": os.path.join(base_path, "cotton dataset"),
    "wheat": os.path.join(base_path, "wheat", "data", "train") # I'll just check train for now
}

total = 0
for crop, path in datasets.items():
    if os.path.exists(path):
        count = sum([len(files) for r, d, files in os.walk(path)])
        print(f"{crop}: {count} images")
        total += count
print(f"Total: {total}")
