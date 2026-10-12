"""
evaluate.py — v3
Evaluates the trained multimodal model (best_model_multimodal.pth) on the test split
and generates comprehensive classification reports and confusion matrices for ALL crops.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from train_multimodal import (
    MultimodalDiseaseNet, CropDiseaseDataset,
    PROJECTV3_PATH, DATASET_DIR, INDEX_PATH, WEATHER_DIR, STATS_PATH, OUTPUTS_DIR,
    DISEASE_MAPS, CLASS_LIST, LABEL_MAP
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model path
    model_path = os.path.join(OUTPUTS_DIR, "best_model_multimodal.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load model
    print(f"Loading model weights from: {model_path}")
    model = MultimodalDiseaseNet(num_classes=len(LABEL_MAP), variant="multimodal").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    test_ds = CropDiseaseDataset(INDEX_PATH, WEATHER_DIR, STATS_PATH, split="test", use_jitter=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_ds)}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, weather, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            weather = weather.to(device)
            outputs = model(images, weather)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 1. Overall Classification Report ──
    unique_labels = sorted(list(set(all_labels) | set(all_preds)))
    target_names = [CLASS_LIST[i] for i in unique_labels]
    
    overall_report = classification_report(
        all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0
    )

    report_path = os.path.join(OUTPUTS_DIR, "classification_report_multimodal.txt")
    with open(report_path, "w") as fp:
        fp.write("Variant: multimodal\n")
        fp.write(f"Test samples: {len(all_labels)}\n\n")
        fp.write("=== OVERALL ===\n")
        fp.write(overall_report)

    print("\n" + "="*60)
    print("OVERALL CLASSIFICATION REPORT")
    print("="*60)
    print(overall_report)

    # ── 2. Per-Class F1 Bar Chart (Overall) ──
    f1_per_class = f1_score(all_labels, all_preds, labels=unique_labels, average=None, zero_division=0)
    plt.figure(figsize=(max(14, len(unique_labels) * 0.55), 5))
    colors = plt.cm.RdYlGn(f1_per_class)
    plt.bar(target_names, f1_per_class, color=colors, edgecolor="black", linewidth=0.5)
    plt.axhline(y=np.mean(f1_per_class), color="navy", linestyle="--", label=f"Mean F1={np.mean(f1_per_class):.3f}")
    plt.xlabel("Class"); plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Score — MULTIMODAL", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
    f1_path = os.path.join(OUTPUTS_DIR, "f1_per_class_multimodal.png")
    plt.savefig(f1_path, dpi=150)
    plt.close()

    # ── 3. Per-Crop Evaluation (Wheat, Rice, Sugarcane, Cotton) ──
    with open(report_path, "a") as fp_report:
        for crop in DISEASE_MAPS.keys():
            crop_class_indices = [i for i, name in enumerate(CLASS_LIST) if name.startswith(f"{crop}_")]
            crop_class_names = [CLASS_LIST[i] for i in crop_class_indices]

            # Filter test samples belonging to this crop
            mask = np.isin(all_labels, crop_class_indices)
            if not np.any(mask):
                print(f"\n[SKIP] No test samples found for crop: {crop}")
                continue

            c_labels = all_labels[mask]
            c_preds = all_preds[mask]

            crop_report = classification_report(
                c_labels, c_preds, labels=crop_class_indices, target_names=crop_class_names, zero_division=0
            )

            print("\n" + "="*60)
            print(f"CROP: {crop.upper()}")
            print("="*60)
            print(crop_report)

            fp_report.write(f"\n=== {crop.upper()} ===\n")
            fp_report.write(crop_report)

            # Confusion Matrix PNG
            cm = confusion_matrix(c_labels, c_preds, labels=crop_class_indices)
            plt.figure(figsize=(max(8, len(crop_class_names) * 1.1), max(6, len(crop_class_names) * 0.9)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[n.replace(f"{crop}_", "") for n in crop_class_names],
                        yticklabels=[n.replace(f"{crop}_", "") for n in crop_class_names])
            plt.title(f"Confusion Matrix — {crop.upper()} (Multimodal)", fontweight="bold")
            plt.xlabel("Predicted Label"); plt.ylabel("True Label")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            cm_path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{crop}_multimodal.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"Saved confusion matrix: {cm_path}")

            # Per-crop F1 Bar Chart
            f1_crop = f1_score(c_labels, c_preds, labels=crop_class_indices, average=None, zero_division=0)
            plt.figure(figsize=(max(8, len(crop_class_names) * 0.8), 4))
            colors_c = plt.cm.RdYlGn(f1_crop)
            short_names = [n.replace(f"{crop}_", "") for n in crop_class_names]
            plt.bar(short_names, f1_crop, color=colors_c, edgecolor="black", linewidth=0.5)
            plt.axhline(y=np.mean(f1_crop), color="navy", linestyle="--", label=f"Mean F1={np.mean(f1_crop):.3f}")
            plt.xlabel("Disease"); plt.ylabel("F1 Score")
            plt.title(f"F1 Score per Disease — {crop.upper()}", fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
            f1c_path = os.path.join(OUTPUTS_DIR, f"f1_{crop}_multimodal.png")
            plt.savefig(f1c_path, dpi=150)
            plt.close()
            print(f"Saved F1 bar chart: {f1c_path}")

    # Also update train_multimodal.py's reporting section so future training runs do this cleanly
    print(f"\nCompleted evaluation! All outputs written to: {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
