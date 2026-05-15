import json
import numpy as np
import os

def verify():
    index_path = "dataset/dataset_index.json"
    if not os.path.exists(index_path):
        print("Index file not found!")
        return
        
    with open(index_path, "r") as f:
        records = json.load(f)
        
    print(f"Total records: {len(records)}")
    
    # Check splits
    splits = {}
    for r in records:
        s = r["split"]
        splits[s] = splits.get(s, 0) + 1
    print(f"Splits: {splits}")
    
    # Check a few npy files
    weather_dir = "dataset/weather"
    npy_files = [f for f in os.listdir(weather_dir) if f.endswith(".npy")]
    print(f"Total weather profiles: {len(npy_files)}")
    
    for i in range(min(5, len(npy_files))):
        path = os.path.join(weather_dir, npy_files[i])
        data = np.load(path)
        print(f"File {npy_files[i]} shape: {data.shape}")
        if data.shape != (30, 11):
            print(f"WARNING: Incorrect shape for {npy_files[i]}")

if __name__ == "__main__":
    verify()
