"""
Plant Disease Detection Model - Training Script (v2 - PyTorch)
==============================================================
Trains EfficientNetV2S models for Rice and Wheat disease classification.
Uses PyTorch which has full native Windows CUDA support.

Key techniques:
- EfficientNetV2S pretrained on ImageNet (torchvision)
- Native Windows GPU via PyTorch CUDA
- Two-phase training: frozen backbone -> fine-tune top 40%
- Cosine Annealing LR scheduler (phase 2)
- Class weight balancing (handles imbalanced datasets)
- Heavy data augmentation
- Label smoothing (0.1)
- Mixed precision (torch.amp) for faster GPU utilisation
- Early stopping, best-checkpoint saving
"""

import os
import sys
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T
from torchvision import datasets, models
from torchvision.models import EfficientNet_V2_S_Weights

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Device setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n[GPU] {gpu_name}  |  VRAM: {vram:.1f} GB")
    print(f"[GPU] CUDA {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
    torch.backends.cudnn.benchmark = True   # faster convolutions
else:
    print("\n[CPU] No CUDA GPU found - training on CPU.")
    print("      Ensure pytorch was installed with: --index-url https://download.pytorch.org/whl/cu126")

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE         = 300       # EfficientNetV2S works great at 300x300
BATCH_SIZE       = 32        # reduce to 16 if you get CUDA OOM
INITIAL_EPOCHS   = 15        # phase 1: classification head only
FINE_TUNE_EPOCHS = 25        # phase 2: top 40% backbone + head
LABEL_SMOOTHING  = 0.1
DROPOUT_RATE     = 0.4
SEED             = 42
NUM_WORKERS      = 0         # set to 0 if you get DataLoader errors on Windows

BASE_DIR   = Path(r"d:\clg\major project\Project folder")
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR  = BASE_DIR / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"\nPyTorch version : {torch.__version__}")
print(f"Device          : {device}")
print(f"Image size      : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(training: bool):
    """Returns augmentation pipeline. Heavier augmentation for training."""
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if training:
        return T.Compose([
            T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            T.RandomCrop((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.1),          # occlusion robustness
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    """
    EfficientNetV2S with custom classification head.
    Phase 1: backbone frozen, head trainable.
    """
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model   = models.efficientnet_v2_s(weights=weights)

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=DROPOUT_RATE / 2),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=DROPOUT_RATE / 4),
        nn.Linear(256, num_classes),
    )

    return model.to(device)


def unfreeze_top_backbone(model, fraction: float = 0.40):
    """
    Unfreeze the top `fraction` of EfficientNetV2S backbone layers
    for fine-tuning. BatchNorm layers stay frozen (transfer learning best practice).
    """
    backbone_layers = list(model.features.children())
    n_unfreeze      = max(1, int(len(backbone_layers) * fraction))
    unfreeze_layers = backbone_layers[-n_unfreeze:]

    for layer in unfreeze_layers:
        for module in layer.modules():
            if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Unfroze top {n_unfreeze}/{len(backbone_layers)} backbone blocks "
          f"({trainable:,}/{total:,} params trainable)")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    """
    Build a WeightedRandomSampler so each class is sampled equally,
    even when the dataset is imbalanced.
    """
    targets = [s[1] for s in dataset.samples]
    counts  = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_class_weights(dataset) -> torch.Tensor:
    """Compute per-class weights for the loss function."""
    targets = np.array([s[1] for s in dataset.samples])
    cw = compute_class_weight('balanced',
                              classes=np.unique(targets),
                              y=targets)
    return torch.tensor(cw, dtype=torch.float32).to(device)


def make_loader(dataset, training: bool, use_sampler: bool = False) -> DataLoader:
    if training and use_sampler:
        sampler = make_weighted_sampler(dataset)
        return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=NUM_WORKERS > 0)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=NUM_WORKERS > 0)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_rice_data():
    """
    Rice_Leaf_AUG/<ClassName>/*.jpg  (6 classes, ~3800 images)
    80/20 train/val split via two ImageFolder calls with different transforms.
    """
    rice_dir = BASE_DIR / "Rice_Leaf_AUG"
    print("\n" + "=" * 60)
    print("  LOADING RICE LEAF DISEASE DATASET  (Rice_Leaf_AUG)")
    print("=" * 60)

    if not rice_dir.exists():
        raise FileNotFoundError(f"Rice dataset not found: {rice_dir}")

    # Use full dataset with train transforms, split manually
    full_train = datasets.ImageFolder(rice_dir, transform=get_transforms(True))
    full_val   = datasets.ImageFolder(rice_dir, transform=get_transforms(False))

    n_total = len(full_train)
    n_val   = int(n_total * 0.2)
    n_train = n_total - n_val

    # Same indices for both (only transforms differ)
    train_idx, val_idx = torch.utils.data.random_split(
        range(n_total), [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    train_sub = torch.utils.data.Subset(full_train, train_idx)
    val_sub   = torch.utils.data.Subset(full_val,   val_idx)

    class_names = full_train.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    counts = Counter([full_train.samples[i][1] for i in train_idx])
    for idx, name in enumerate(class_names):
        print(f"  {name}: {counts[idx]} train samples")
    print(f"Train: {n_train}  |  Val: {n_val}")

    # Class weights from training subset
    train_targets = np.array([full_train.samples[i][1] for i in train_idx])
    cw = compute_class_weight('balanced',
                              classes=np.unique(train_targets),
                              y=train_targets)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)

    return train_loader, val_loader, class_names, class_weights


def load_wheat_data():
    """Wheat dataset with pre-defined train/valid/test split folders."""
    wheat_dir = BASE_DIR / "wheat" / "data"
    print("\n" + "=" * 60)
    print("  LOADING WHEAT PLANT DISEASE DATASET")
    print("=" * 60)

    train_ds = datasets.ImageFolder(wheat_dir / "train", transform=get_transforms(True))
    val_ds   = datasets.ImageFolder(wheat_dir / "valid", transform=get_transforms(False))
    test_ds  = datasets.ImageFolder(wheat_dir / "test",  transform=get_transforms(False))

    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    counts = Counter([s[1] for s in train_ds.samples])
    for idx, name in enumerate(class_names):
        print(f"  {name}: {counts[idx]} samples")

    class_weights = get_class_weights(train_ds)

    train_loader = make_loader(train_ds, training=True,  use_sampler=True)
    val_loader   = make_loader(val_ds,   training=False)
    test_loader  = make_loader(test_ds,  training=False)

    return train_loader, val_loader, test_loader, class_names, class_weights


# ── Training Loop ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_acc: float, model) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False   # don't stop
        self.counter += 1
        return self.counter >= self.patience   # stop


def run_epoch(model, loader, criterion, optimizer, scaler,
              training: bool, use_amp: bool = False) -> tuple[float, float]:
    """Single epoch — returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def train_model(train_loader, val_loader, class_names, class_weights,
                crop_name: str):
    """
    Two-phase training:
      Phase 1 — head only (frozen backbone) with ReduceLROnPlateau
      Phase 2 — top 40% backbone + head with CosineAnnealingLR
    """
    num_classes = len(class_names)
    model       = build_model(num_classes)
    criterion   = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=LABEL_SMOOTHING)
    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    checkpoint_path = MODELS_DIR / f"{crop_name}_best_model.pt"

    # ── Phase 1: head only ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Training Classification Head ({crop_name.upper()})")
    print(f"{'=' * 60}")

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    stopper1   = EarlyStopping(patience=6)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(1, INITIAL_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer1, scaler, training=True,
                                    use_amp=use_amp)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer1, scaler, training=False,
                                    use_amp=use_amp)
        scheduler1.step(va_acc)

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        lr = optimizer1.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{INITIAL_EPOCHS} | "
              f"loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {va_loss:.4f} val_acc: {va_acc:.4f} | "
              f"lr: {lr:.2e} | {time.time()-t0:.0f}s")

        if stopper1(va_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(stopper1.best_state)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[Phase 1] Best Val Accuracy: {stopper1.best_acc:.4f} "
          f"({stopper1.best_acc * 100:.2f}%)")

    # ── Phase 2: unfreeze top 40% backbone ────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Fine-tuning Top 40% of EfficientNetV2S ({crop_name.upper()})")
    print(f"{'=' * 60}")

    unfreeze_top_backbone(model, fraction=0.40)

    optimizer2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=FINE_TUNE_EPOCHS, eta_min=1e-7
    )
    stopper2   = EarlyStopping(patience=8)

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer2, scaler, training=True,
                                    use_amp=use_amp)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer2, scaler, training=False,
                                    use_amp=use_amp)
        scheduler2.step()

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        lr = optimizer2.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{FINE_TUNE_EPOCHS} | "
              f"loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {va_loss:.4f} val_acc: {va_acc:.4f} | "
              f"lr: {lr:.2e} | {time.time()-t0:.0f}s")

        if stopper2(va_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(stopper2.best_state)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[Phase 2] Best Val Accuracy: {stopper2.best_acc:.4f} "
          f"({stopper2.best_acc * 100:.2f}%)")

    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, class_names: list, tag: str) -> float:
    """Full evaluation: accuracy + classification report + confusion matrix."""
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {tag.upper()}")
    print(f"{'=' * 60}")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    print(f"\n[RESULT] Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    # Save text report
    report_path = MODELS_DIR / f"{tag}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"  {tag.upper()} — EfficientNetV2S Classification Report\n")
        f.write(f"  Input: {IMG_SIZE}x{IMG_SIZE}  |  PyTorch {torch.__version__}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(10, len(class_names) * 0.9)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{tag.title()} — Confusion Matrix (EfficientNetV2S)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = PLOTS_DIR / f"{tag}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f" Confusion matrix saved to: {cm_path}")

    return accuracy


# ── Plot History ──────────────────────────────────────────────────────────────

def plot_training_history(history: dict, crop_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(history['train_acc']) + 1)

    ax1.plot(epochs, history['train_acc'], label='Train Accuracy',
             linewidth=2, color='#2196F3')
    ax1.plot(epochs, history['val_acc'],   label='Val Accuracy',
             linewidth=2, color='#FF5722')
    ax1.axvline(x=INITIAL_EPOCHS, color='gray', linestyle='--', alpha=0.7)
    ax1.text(INITIAL_EPOCHS + 0.3, 0.05, 'Fine-tune\nstart', fontsize=9, color='gray')
    ax1.set_title(f'{crop_name.title()} — Accuracy (EfficientNetV2S)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2.plot(epochs, history['train_loss'], label='Train Loss',
             linewidth=2, color='#2196F3')
    ax2.plot(epochs, history['val_loss'],   label='Val Loss',
             linewidth=2, color='#FF5722')
    ax2.axvline(x=INITIAL_EPOCHS, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title(f'{crop_name.title()} — Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'{crop_name.title()} Plant Disease Detection\n'
        f'EfficientNetV2S | {IMG_SIZE}x{IMG_SIZE} | PyTorch CUDA',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{crop_name}_training_history.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Training plot saved to: {plot_path}")


# ── Save Artifacts ────────────────────────────────────────────────────────────

def save_model_artifacts(model, class_names: list, crop_name: str,
                         accuracy: float):
    # Save full model state dict
    model_path = MODELS_DIR / f"{crop_name}_disease_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'img_size': IMG_SIZE,
        'architecture': 'EfficientNetV2S',
    }, model_path)
    print(f" Model saved to: {model_path}")

    # Class labels JSON (keep compatible with existing predict.py format)
    labels_path = MODELS_DIR / f"{crop_name}_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({str(i): name for i, name in enumerate(class_names)},
                  f, indent=2)
    print(f" Class labels saved to: {labels_path}")

    # Model info
    info = {
        "crop": crop_name,
        "framework": "PyTorch",
        "architecture": "EfficientNetV2S (torchvision pretrained on ImageNet)",
        "input_size": IMG_SIZE,
        "num_classes": len(class_names),
        "class_names": class_names,
        "validation_accuracy": float(accuracy),
        "techniques": [
            "Transfer Learning (EfficientNetV2S ImageNet weights)",
            "PyTorch native Windows CUDA",
            "Mixed Precision (torch.amp autocast + GradScaler)",
            "Two-phase training: frozen head -> top-40% fine-tune",
            "Cosine Annealing LR (phase 2) + ReduceLROnPlateau (phase 1)",
            "WeightedRandomSampler + CrossEntropy class weights",
            "Label Smoothing (0.1)",
            "Heavy augmentation (flip, rotate, jitter, erase)",
            "Gradient clipping (max_norm=1.0)",
            "BatchNorm frozen during fine-tuning",
            "EarlyStopping with best-weight restoration",
        ],
    }
    info_path = MODELS_DIR / f"{crop_name}_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f" Model info saved to: {info_path}")


# ── Crop Pipelines ────────────────────────────────────────────────────────────

def train_rice() -> float:
    print("\n" + "=" * 60)
    print("  RICE LEAF DISEASE DETECTION  (EfficientNetV2S + PyTorch)")
    print("=" * 60)
    train_loader, val_loader, class_names, class_weights = load_rice_data()
    model, history = train_model(train_loader, val_loader,
                                 class_names, class_weights, "rice")
    accuracy = evaluate_model(model, val_loader, class_names, "rice")
    plot_training_history(history, "rice")
    save_model_artifacts(model, class_names, "rice", accuracy)
    return accuracy


def train_wheat():
    print("\n" + "=" * 60)
    print("  WHEAT PLANT DISEASE DETECTION  (EfficientNetV2S + PyTorch)")
    print("=" * 60)
    train_loader, val_loader, test_loader, class_names, class_weights = load_wheat_data()
    model, history = train_model(train_loader, val_loader,
                                 class_names, class_weights, "wheat")
    val_accuracy  = evaluate_model(model, val_loader,  class_names, "wheat")
    print(f"\n{'=' * 60}\n  TEST SET EVALUATION: WHEAT\n{'=' * 60}")
    test_accuracy = evaluate_model(model, test_loader, class_names, "wheat_test")
    plot_training_history(history, "wheat")
    save_model_artifacts(model, class_names, "wheat", val_accuracy)
    return val_accuracy, test_accuracy


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PLANT DISEASE DETECTION — PyTorch Training Pipeline v2")
    print("=" * 60)
    print(f"  Backbone         : EfficientNetV2S")
    print(f"  Image size       : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Phase 1 epochs   : {INITIAL_EPOCHS}")
    print(f"  Phase 2 epochs   : {FINE_TUNE_EPOCHS}")
    print(f"  Label smoothing  : {LABEL_SMOOTHING}")
    print(f"  Fine-tune layers : top 40% of backbone")
    print(f"  Device           : {device}")
    print("=" * 60)

    rice_accuracy = train_rice()
    wheat_val_acc, wheat_test_acc = train_wheat()

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Rice  -- Val Accuracy  : {rice_accuracy * 100:.2f}%")
    print(f"  Wheat -- Val Accuracy  : {wheat_val_acc * 100:.2f}%")
    print(f"  Wheat -- Test Accuracy : {wheat_test_acc * 100:.2f}%")
    print("=" * 60)
    print(f"\n  Models saved to : {MODELS_DIR}")
    print(f"  Plots saved to  : {PLOTS_DIR}")
    print("=" * 60)
