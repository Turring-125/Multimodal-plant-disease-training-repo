"""
train_multimodal.py — v3

Key changes from v2:
  - WeatherEncoder operates on 7-timestep sequences (was 30)
  - WeatherEncoder.forward() can return attention weights for XAI
  - Weather jitter augmentation (Part 8B.2 of INSTRUCTIONS_v3)
  - Weather file path uses 'weather_file' key from dataset_index.json
  - All other training logic (mixup, early stopping, cosine schedule) retained
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import time
import copy
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from PIL import Image
import numpy as np
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECTV3_PATH = r"d:\clg\major project\Project folder\projectv3"
DATASET_DIR    = os.path.join(PROJECTV3_PATH, "dataset")
INDEX_PATH     = os.path.join(DATASET_DIR, "dataset_index.json")
WEATHER_DIR    = os.path.join(DATASET_DIR, "weather")
STATS_PATH     = os.path.join(DATASET_DIR, "weather_stats.npy")

# All training outputs (weights, plots, reports) go here
OUTPUTS_DIR    = os.path.join(PROJECTV3_PATH, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_DAYS = 7
N_FEATURES  = 11
IMG_SIZE    = 300

# ── Disease label maps ────────────────────────────────────────────────────────

DISEASE_MAPS = {
    "wheat": {
        "Aphid": 0, "Black Rust": 1, "Blast": 2, "Brown Rust": 3,
        "Common Root Rot": 4, "Fusarium Head Blight": 5, "Healthy": 6,
        "Leaf Blight": 7, "Mildew": 8, "Mite": 9, "Septoria": 10,
        "Smut": 11, "Stem Fly": 12, "Tan Spot": 13, "Yellow Rust": 14,
    },
    "rice": {
        "Bacterial Leaf Blight": 0, "Brown Spot": 1, "Healthy Rice Leaf": 2,
        "Leaf Blast": 3, "Leaf Scald": 4, "Sheath Blight": 5,
    },
    "sugarcane": {
        "Healthy": 0, "Mosaic": 1, "Redrot": 2, "Rust": 3, "Yellow Disease": 4,
    },
    "cotton": {
        "Bacterial Blight": 0, "Curl Virus": 1, "Healthy Leaf": 2,
        "Herbicide Damage": 3, "Leaf Hopper Jassids": 4,
        "Leaf Reddening": 5, "Leaf Variegation": 6,
    },
}

# Unified label map across all crops
CLASS_LIST = []
for crop, m in DISEASE_MAPS.items():
    for disease in m.keys():
        CLASS_LIST.append(f"{crop}_{disease}")
LABEL_MAP = {name: i for i, name in enumerate(CLASS_LIST)}

# ── Augmentations ─────────────────────────────────────────────────────────────

def jitter_weather(wx, sigma=0.03):
    """
    Add small Gaussian noise to normalized weather tensor.
    Applied ONLY at train time, after normalization.
    Sigma ~0.03 blurs exact synthetic signatures without erasing seasonal shape.
    Reference: Um et al., 2017 (ICMI) — Part 8B.2 of INSTRUCTIONS_v3.
    """
    return wx + torch.randn_like(wx) * sigma


# ── Model components ──────────────────────────────────────────────────────────

class WeatherEncoder(nn.Module):
    """
    1D CNN + Multi-head Self-Attention encoder for 7-day weather sequences.

    Input shape: [B, 7, 11]
    Output shape: [B, out_dim]

    When return_attention=True, also returns attention weights [B, 7, 7].
    """
    def __init__(self, n_features=N_FEATURES, seq_len=WINDOW_DAYS, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 4-head attention over 7 timesteps
        self.attn = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True, dropout=0.1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, out_dim)
        self.norm = nn.LayerNorm(128)

    def forward(self, x, return_attention=False):
        # x: [B, 7, 11]
        x = x.permute(0, 2, 1)                    # [B, 11, 7]
        x = self.conv(x)                            # [B, 128, 7]
        x = x.permute(0, 2, 1)                    # [B, 7, 128]
        attn_out, attn_weights = self.attn(x, x, x,
                                           need_weights=True,
                                           average_attn_weights=True)
        # attn_weights: [B, 7, 7]
        x = self.norm(x + attn_out)               # residual + LayerNorm
        x = x.permute(0, 2, 1)                    # [B, 128, 7]
        x = self.pool(x).squeeze(-1)              # [B, 128]
        feat = self.fc(x)                          # [B, out_dim]
        if return_attention:
            return feat, attn_weights
        return feat


class MultimodalDiseaseNet(nn.Module):
    """
    Dual-stream: EfficientNetV2-S (image) + 1D CNN+Attention (weather).
    Supports three variants: 'multimodal', 'image-only', 'weather-only'.
    """
    def __init__(self, num_classes, weather_dim=128, img_dim=512,
                 variant="multimodal"):
        super().__init__()
        self.variant = variant

        if variant in ("multimodal", "image-only"):
            self.image_encoder = timm.create_model(
                "tf_efficientnetv2_s.in1k", pretrained=True, num_classes=0
            )  # → [B, 1280]
            self.image_fc = nn.Sequential(
                nn.Linear(1280, img_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
            )

        if variant in ("multimodal", "weather-only"):
            self.weather_encoder = WeatherEncoder(
                n_features=N_FEATURES, seq_len=WINDOW_DAYS, out_dim=weather_dim
            )

        if variant == "multimodal":
            self.classifier = nn.Sequential(
                nn.Linear(img_dim + weather_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
        elif variant == "image-only":
            self.classifier = nn.Sequential(
                nn.Linear(img_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )
        else:  # weather-only
            self.classifier = nn.Sequential(
                nn.Linear(weather_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

    def forward(self, image, weather, return_attention=False):
        if self.variant == "multimodal":
            img_feat = self.image_fc(self.image_encoder(image))
            if return_attention:
                wx_feat, attn_w = self.weather_encoder(weather, return_attention=True)
                fused = torch.cat([img_feat, wx_feat], dim=1)
                return self.classifier(fused), attn_w
            else:
                wx_feat = self.weather_encoder(weather)
                fused   = torch.cat([img_feat, wx_feat], dim=1)
                return self.classifier(fused)

        elif self.variant == "image-only":
            img_feat = self.image_fc(self.image_encoder(image))
            return self.classifier(img_feat)

        else:  # weather-only
            if return_attention:
                wx_feat, attn_w = self.weather_encoder(weather, return_attention=True)
                return self.classifier(wx_feat), attn_w
            wx_feat = self.weather_encoder(weather)
            return self.classifier(wx_feat)


# ── Robustness helpers ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def unfreeze_backbone(model, fraction=0.40):
    """Unfreeze top `fraction` of EfficientNetV2 conv blocks for fine-tuning."""
    if not hasattr(model, "image_encoder"):
        return
    backbone = model.image_encoder
    for param in backbone.parameters():
        param.requires_grad = False

    if hasattr(backbone, "blocks"):
        blocks = list(backbone.blocks.children())
        n_unfreeze = max(1, int(len(blocks) * fraction))
        for block in blocks[-n_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
        for name in ("conv_head", "bn2"):
            if hasattr(backbone, name):
                for param in getattr(backbone, name).parameters():
                    param.requires_grad = True
    else:
        all_params = list(backbone.named_parameters())
        n_unfreeze = max(1, int(len(all_params) * fraction))
        for _, param in all_params[-n_unfreeze:]:
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Unfroze top {fraction*100:.0f}% of backbone ({trainable:,} params trainable)")


def mixup_data(x_img, x_wx, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x_img.size(0), device=x_img.device)
    mixed_img = lam * x_img + (1 - lam) * x_img[idx]
    mixed_wx  = lam * x_wx  + (1 - lam) * x_wx[idx]
    return mixed_img, mixed_wx, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    def __init__(self, index_path, weather_dir, stats_path,
                 split="train", transform=None, use_jitter=True):
        with open(index_path) as f:
            all_records = json.load(f)
        self.records     = [r for r in all_records if r["split"] == split]
        self.weather_dir = weather_dir
        self.split       = split
        self.use_jitter  = use_jitter and (split == "train")
        self.transform   = transform or self._default_transform(split)

        stats         = np.load(stats_path, allow_pickle=True).item()
        self.wx_mean  = torch.tensor(stats["mean"], dtype=torch.float32)  # [11]
        self.wx_std   = torch.tensor(stats["std"],  dtype=torch.float32)  # [11]

    def _default_transform(self, split):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return T.Compose([
            T.Resize(int(IMG_SIZE * 1.1)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        img = self.transform(img)

        wf  = r.get("weather_file", f"{r['id']}_weather.npy")
        wx  = np.load(os.path.join(self.weather_dir, wf)).astype(np.float32)
        wx  = torch.tensor(wx)                            # [7, 11]
        wx  = (wx - self.wx_mean) / (self.wx_std + 1e-8) # z-score normalize

        if self.use_jitter:
            wx = jitter_weather(wx, sigma=0.03)

        label_name = f"{r['crop_type']}_{r['disease_label']}"
        label      = LABEL_MAP[label_name]

        return img, wx, label

    def get_class_weights(self):
        from collections import Counter
        labels = [LABEL_MAP[f"{r['crop_type']}_{r['disease_label']}"]
                  for r in self.records]
        counts = Counter(labels)
        n_total = len(labels)
        weights = []
        for i in range(len(LABEL_MAP)):
            c = counts.get(i, 0)
            weights.append(n_total / (len(counts) * c) if c > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float32)


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, variant,
                    scaler=None, use_mixup=True):
    model.train()
    running_loss = correct = total = 0

    for images, weather, labels in tqdm(loader, desc="Train", leave=False):
        images  = images.to(device, non_blocking=True)
        weather = weather.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        if use_mixup:
            images, weather, la, lb, lam = mixup_data(images, weather, labels)

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast(device_type="cuda" if device.type == "cuda" else "cpu")
        with ctx:
            outputs = model(images, weather)
            loss = (mixup_criterion(criterion, outputs, la, lb, lam)
                    if use_mixup else criterion(outputs, labels))

        if scaler:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total  += labels.size(0)
        if use_mixup:
            correct += (lam * pred.eq(la).sum().item() +
                        (1 - lam) * pred.eq(lb).sum().item())
        else:
            correct += pred.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, scaler=None):
    model.eval()
    running_loss = correct = total = 0

    for images, weather, labels in loader:
        images  = images.to(device, non_blocking=True)
        weather = weather.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        ctx = autocast(device_type="cuda" if device.type == "cuda" else "cpu")
        with ctx:
            outputs = model(images, weather)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total  += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return running_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train multimodal crop disease model v3")
    parser.add_argument("--variant",    default="multimodal",
                        choices=["multimodal", "image-only", "weather-only"])
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup",     type=int, default=5)
    parser.add_argument("--no_mixup",   action="store_true")
    parser.add_argument("--no_jitter",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    use_jitter = not args.no_jitter
    train_ds = CropDiseaseDataset(INDEX_PATH, WEATHER_DIR, STATS_PATH,
                                  split="train", use_jitter=use_jitter)
    val_ds   = CropDiseaseDataset(INDEX_PATH, WEATHER_DIR, STATS_PATH,
                                  split="val",   use_jitter=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = MultimodalDiseaseNet(
        num_classes=len(LABEL_MAP), variant=args.variant
    ).to(device)

    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler    = GradScaler("cuda") if device.type == "cuda" else None

    early_stopping = EarlyStopping(patience=8)
    best_val_acc   = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Phase 1: Warmup ────────────────────────────────────────────────────
    print(f"\n── Phase 1: Warmup ({args.warmup} epochs) ──")
    if hasattr(model, "image_encoder"):
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    for epoch in range(args.warmup):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, args.variant, scaler,
                                  use_mixup=not args.no_mixup)
        vl, va = validate(model, val_loader, criterion, device, scaler)
        scheduler.step(va)

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"  Warmup {epoch+1}/{args.warmup} | "
              f"train_loss={tl:.4f} acc={ta:.4f} | "
              f"val_loss={vl:.4f} acc={va:.4f}")

    # ── Phase 2: Fine-tuning ───────────────────────────────────────────────
    print(f"\n── Phase 2: Fine-tuning ({args.epochs} epochs) ──")
    unfreeze_backbone(model, fraction=0.40)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    for epoch in range(args.epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, args.variant, scaler,
                                  use_mixup=not args.no_mixup)
        vl, va = validate(model, val_loader, criterion, device, scaler)
        scheduler.step()

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"  FT {epoch+1}/{args.epochs} | "
              f"train_loss={tl:.4f} acc={ta:.4f} | "
              f"val_loss={vl:.4f} acc={va:.4f}")

        model_save_path = os.path.join(OUTPUTS_DIR,
                                       f"best_model_{args.variant}.pth")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), model_save_path)
            print(f"    ✓ Best model saved (val_acc={va:.4f})")

        if early_stopping(va, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if early_stopping.best_state:
        model.load_state_dict(early_stopping.best_state)

    # ── Training history plot ──────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"],   label="Val Acc")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(OUTPUTS_DIR, f"training_history_{args.variant}.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Training history plot saved: {hist_path}")

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    test_ds = CropDiseaseDataset(INDEX_PATH, WEATHER_DIR, STATS_PATH,
                                  split="test", use_jitter=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, weather, labels in tqdm(test_loader, desc="Testing"):
            images  = images.to(device)
            weather = weather.to(device)
            out     = model(images, weather)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Overall report — save to file ────────────────────────────────────────
    unique_labels = sorted(set(all_labels) | set(all_preds))
    target_names  = [CLASS_LIST[i] for i in unique_labels]
    overall_report = classification_report(all_labels, all_preds,
                                           labels=unique_labels,
                                           target_names=target_names)
    print("\nOverall Classification Report:")
    print(overall_report)

    report_path = os.path.join(OUTPUTS_DIR,
                               f"classification_report_{args.variant}.txt")
    with open(report_path, "w") as fp:
        fp.write(f"Variant: {args.variant}\n")
        fp.write(f"Test samples: {len(all_labels)}\n\n")
        fp.write("=== OVERALL ===\n")
        fp.write(overall_report)
    print(f"Classification report saved: {report_path}")

    # ── Per-class F1 bar chart (overall) ────────────────────────────────────
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(all_labels, all_preds,
                             labels=unique_labels, average=None, zero_division=0)
    plt.figure(figsize=(max(14, len(unique_labels) * 0.55), 5))
    colors = plt.cm.RdYlGn(f1_per_class)  # red=low, green=high
    plt.bar(target_names, f1_per_class, color=colors, edgecolor="black",
            linewidth=0.5)
    plt.axhline(y=np.mean(f1_per_class), color="navy", linestyle="--",
                label=f"Mean F1={np.mean(f1_per_class):.3f}")
    plt.xlabel("Class"); plt.ylabel("F1 Score")
    plt.title(f"Per-Class F1 Score — {args.variant.upper()}",
              fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
    f1_path = os.path.join(OUTPUTS_DIR, f"f1_per_class_{args.variant}.png")
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"F1 bar chart saved: {f1_path}")

    # ── Per-crop confusion matrix + per-crop report ──────────────────────────
    with open(report_path, "a") as fp_report:
        for crop in DISEASE_MAPS:
            crop_idx = [i for i, n in enumerate(CLASS_LIST) if n.startswith(crop)]
            c_labels = [all_labels[i] for i, l in enumerate(all_labels)
                        if l in crop_idx]
            c_preds  = [all_preds[i]  for i, l in enumerate(all_labels)
                        if l in crop_idx]
            if not c_labels:
                continue
            unique_c = sorted(set(c_labels))
            names_c  = [CLASS_LIST[i] for i in unique_c]
            remap    = {old: new for new, old in enumerate(unique_c)}
            c_labels_r = [remap[l] for l in c_labels]
            c_preds_r  = [remap.get(p, -1) for p in c_preds]

            crop_report = classification_report(c_labels_r, c_preds_r,
                                                labels=list(range(len(names_c))),
                                                target_names=names_c,
                                                zero_division=0)
            print(f"\n── {crop.upper()} ──")
            print(crop_report)
            fp_report.write(f"\n=== {crop.upper()} ===\n")
            fp_report.write(crop_report)

            # Confusion matrix PNG
            cm = confusion_matrix(c_labels_r, c_preds_r, labels=list(range(len(names_c))))
            plt.figure(figsize=(max(8, len(names_c)), max(6, len(names_c))))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=names_c, yticklabels=names_c)
            plt.title(f"Confusion Matrix — {crop.upper()}")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            cm_path = os.path.join(OUTPUTS_DIR,
                                   f"confusion_matrix_{crop}_{args.variant}.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"  Confusion matrix saved: {os.path.basename(cm_path)}")

            # Per-crop F1 bar chart
            f1_crop = f1_score(c_labels_r, c_preds_r, average=None, zero_division=0)
            plt.figure(figsize=(max(8, len(names_c) * 0.7), 4))
            colors_c = plt.cm.RdYlGn(f1_crop)
            plt.bar(names_c, f1_crop, color=colors_c, edgecolor="black",
                    linewidth=0.5)
            plt.axhline(y=np.mean(f1_crop), color="navy", linestyle="--",
                        label=f"Mean={np.mean(f1_crop):.3f}")
            plt.xlabel("Disease"); plt.ylabel("F1 Score")
            plt.title(f"F1 per Disease — {crop.upper()}", fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
            f1c_path = os.path.join(OUTPUTS_DIR,
                                    f"f1_{crop}_{args.variant}.png")
            plt.savefig(f1c_path, dpi=150)
            plt.close()

    print(f"\nAll outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
