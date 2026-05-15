import numpy as np
import json
import os
import torch

def compute_stats(index_path, weather_dir):
    with open(index_path) as f:
        records = json.load(f)
    
    train_records = [r for r in records if r['split'] == 'train']
    print(f"Computing stats for {len(train_records)} training samples...")
    
    all_wx = []
    for r in train_records:
        path = os.path.join(weather_dir, f"profile_{r['profile_id']}.npy")
        if os.path.exists(path):
            all_wx.append(np.load(path))
    
    all_wx = np.stack(all_wx) # [N, 30, 11]
    mean = np.mean(all_wx, axis=(0, 1))
    std = np.std(all_wx, axis=(0, 1))
    
    stats = {"mean": mean, "std": std}
    np.save("dataset/weather_stats.npy", stats)
    print("Stats saved to dataset/weather_stats.npy")
    print("Mean:", mean)
    print("Std:", std)

if __name__ == "__main__":
    compute_stats("dataset/dataset_index.json", "dataset/weather")
