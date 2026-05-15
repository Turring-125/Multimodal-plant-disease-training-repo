import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import copy
import random

# --- CONFIGURATION ---
IMG_SIZE = 300
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

# Create a unified label mapping for all 33 classes
CLASS_LIST = []
for crop, m in DISEASE_MAPS.items():
    for disease in m.keys():
        CLASS_LIST.append(f"{crop}_{disease}")
LABEL_MAP = {name: i for i, name in enumerate(CLASS_LIST)}

# --- ROBUSTNESS COMPONENTS ---

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = 0.0
        self.counter = 0
        self.best_state = None

    def __call__(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

def unfreeze_backbone(model, fraction=0.40):
    """Unfreeze the top fraction of the image encoder's conv blocks."""
    if hasattr(model, 'image_encoder'):
        backbone = model.image_encoder
        
        # Freeze everything first
        for param in backbone.parameters():
            param.requires_grad = False
        
        # EfficientNetV2 stores conv blocks inside backbone.blocks (Sequential of 6 blocks)
        if hasattr(backbone, 'blocks'):
            blocks = list(backbone.blocks.children())
            n_blocks = len(blocks)
            n_unfreeze = max(1, int(n_blocks * fraction))
            
            for block in blocks[-n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            # Also unfreeze conv_head and bn2 (post-block layers)
            for name in ['conv_head', 'bn2']:
                if hasattr(backbone, name):
                    for param in getattr(backbone, name).parameters():
                        param.requires_grad = True
        else:
            # Fallback: unfreeze last fraction of all named parameters
            all_params = list(backbone.named_parameters())
            n_unfreeze_params = max(1, int(len(all_params) * fraction))
            for name, param in all_params[-n_unfreeze_params:]:
                param.requires_grad = True
            n_blocks = len(all_params)
            n_unfreeze = n_unfreeze_params
        
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Unfroze top {n_unfreeze}/{n_blocks} blocks ({trainable:,} params trainable)")


def mixup_data(x_img, x_wx, y, alpha=0.2):
    """Apply Mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x_img.size(0)
    index = torch.randperm(batch_size, device=x_img.device)
    mixed_img = lam * x_img + (1 - lam) * x_img[index]
    mixed_wx  = lam * x_wx  + (1 - lam) * x_wx[index]
    y_a, y_b = y, y[index]
    return mixed_img, mixed_wx, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- MODEL COMPONENTS ---

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
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.1)
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
        return self.fc(x)

class MultimodalDiseaseNet(nn.Module):
    def __init__(self, num_classes, weather_dim=128, img_dim=512, variant='multimodal'):
        super().__init__()
        self.variant = variant
        
        if variant in ['multimodal', 'image-only']:
            self.image_encoder = timm.create_model('tf_efficientnetv2_s.in1k', pretrained=True, num_classes=0)
            self.image_fc = nn.Sequential(
                nn.Linear(1280, img_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            
        if variant in ['multimodal', 'weather-only']:
            self.weather_encoder = WeatherEncoder(n_features=11, out_dim=weather_dim)
            
        if variant == 'multimodal':
            self.classifier = nn.Sequential(
                nn.Linear(img_dim + weather_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif variant == 'image-only':
            self.classifier = nn.Sequential(
                nn.Linear(img_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
        else: # weather-only
            self.classifier = nn.Sequential(
                nn.Linear(weather_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

    def forward(self, image, weather):
        if self.variant == 'multimodal':
            img_feat = self.image_fc(self.image_encoder(image))
            wx_feat  = self.weather_encoder(weather)
            fused    = torch.cat([img_feat, wx_feat], dim=1)
            return self.classifier(fused)
        elif self.variant == 'image-only':
            img_feat = self.image_fc(self.image_encoder(image))
            return self.classifier(img_feat)
        else: # weather-only
            wx_feat = self.weather_encoder(weather)
            return self.classifier(wx_feat)

# --- DATASET ---

class CropDiseaseDataset(Dataset):
    def __init__(self, index_path, weather_dir, stats_path, split="train", transform=None):
        with open(index_path) as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r["split"] == split]
        self.weather_dir = weather_dir
        self.transform   = transform or self._default_transform(split)
        
        stats = np.load(stats_path, allow_pickle=True).item()
        self.wx_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.wx_std  = torch.tensor(stats["std"],  dtype=torch.float32)

    def _default_transform(self, split):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return T.Compose([
            T.Resize(int(IMG_SIZE * 1.1)), T.CenterCrop(IMG_SIZE), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        img = self.transform(img)
        
        wx = np.load(os.path.join(self.weather_dir, f"profile_{r['profile_id']}.npy"))
        wx = torch.tensor(wx, dtype=torch.float32)
        wx = (wx - self.wx_mean) / (self.wx_std + 1e-8)
        
        label_name = f"{r['crop_type']}_{r['disease_label']}"
        label = LABEL_MAP[label_name]
        
        return img, wx, label

    def get_class_weights(self):
        from collections import Counter
        labels = [LABEL_MAP[f"{r['crop_type']}_{r['disease_label']}"] for r in self.records]
        counts = Counter(labels)
        n_total = len(labels)
        # Handle missing classes in split if any
        weights = []
        for i in range(len(LABEL_MAP)):
            count = counts.get(i, 0)
            if count > 0:
                weights.append(n_total / (len(counts) * count))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

# --- TRAINING ENGINE ---

def train_one_epoch(model, loader, optimizer, criterion, device, variant, scaler=None, use_mixup=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, weather, labels in pbar:
        images, weather, labels = images.to(device, non_blocking=True), weather.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Apply Mixup if enabled
        if use_mixup:
            images, weather, labels_a, labels_b, lam = mixup_data(images, weather, labels, alpha=0.2)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
            with autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                outputs = model(images, weather)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, weather)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if use_mixup:
            correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device, variant, scaler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, weather, labels in loader:
        images, weather, labels = images.to(device, non_blocking=True), weather.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        if scaler:
            with autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                outputs = model(images, weather)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images, weather)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='multimodal', choices=['multimodal', 'image-only', 'weather-only'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_ds = CropDiseaseDataset("dataset/dataset_index.json", "dataset/weather", "dataset/weather_stats.npy", split="train")
    val_ds   = CropDiseaseDataset("dataset/dataset_index.json", "dataset/weather", "dataset/weather_stats.npy", split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = MultimodalDiseaseNet(num_classes=len(LABEL_MAP), variant=args.variant).to(device)
    
    # Loss & Optimizer
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Mixed Precision Setup
    scaler = GradScaler("cuda") if "cuda" in str(device) else None
    
    # Optimizer for Phase 1 (Warmup)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    early_stopping = EarlyStopping(patience=8)
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nPHASE 1: Warmup ({args.warmup} epochs)")
    for epoch in range(args.warmup):
        print(f"\nWarmup Epoch {epoch+1}/{args.warmup}")
        # Freeze image encoder
        if hasattr(model, 'image_encoder'):
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.variant, scaler, use_mixup=True)
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.variant, scaler)
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f}")

    # PHASE 2: Fine-Tuning
    print(f"\nPHASE 2: Fine-Tuning ({args.epochs} epochs)")
    unfreeze_backbone(model, fraction=0.40)
    
    # Re-initialize optimizer for Phase 2 — only trainable params
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    for epoch in range(args.epochs):
        print(f"\nFine-Tune Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.variant, scaler, use_mixup=True)
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.variant, scaler)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{args.variant}.pth")
            print(f"  Best model saved! (Acc: {best_val_acc:.4f})")
            
        if early_stopping(val_acc, model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if early_stopping.best_state:
        model.load_state_dict(early_stopping.best_state)

    # Save training history plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig(f'training_history_{args.variant}.png')
    plt.close()

    # Evaluation on Test Set
    print("\nTraining finished. Evaluating on Test Set...")
    test_ds = CropDiseaseDataset("dataset/dataset_index.json", "dataset/weather", "dataset/weather_stats.npy", split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model.load_state_dict(torch.load(f"best_model_{args.variant}.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, weather, labels in test_loader:
            images, weather, labels = images.to(device), weather.to(device), labels.to(device)
            outputs = model(images, weather)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nClassification Report:")
    unique_labels_present = sorted(list(set(all_labels) | set(all_preds)))
    present_target_names = [CLASS_LIST[i] for i in unique_labels_present]
    print(classification_report(all_labels, all_preds, labels=unique_labels_present, target_names=present_target_names))
    
    # Per-crop classification report
    for crop in DISEASE_MAPS.keys():
        print(f"\n--- {crop.upper()} Classification Report ---")
        crop_indices = [i for i, label_name in enumerate(CLASS_LIST) if label_name.startswith(crop)]
        crop_labels = [all_labels[i] for i, l in enumerate(all_labels) if l in crop_indices]
        crop_preds = [all_preds[i] for i, l in enumerate(all_labels) if l in crop_indices]
        
        if crop_labels:
            unique_labels = sorted(list(set(crop_labels)))
            target_names = [CLASS_LIST[i] for i in unique_labels]
            # Map labels to relative indices for report
            label_remap = {old: new for new, old in enumerate(unique_labels)}
            crop_labels_remap = [label_remap[l] for l in crop_labels]
            crop_preds_remap = [label_remap.get(p, -1) for p in crop_preds] # Handle OOD preds
            
            print(classification_report(crop_labels_remap, crop_preds_remap, target_names=target_names))
            
            # Save Confusion Matrix Heatmap
            cm = confusion_matrix(crop_labels_remap, crop_preds_remap)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.title(f'Confusion Matrix - {crop.upper()}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{crop}_{args.variant}.png')
            plt.close()

if __name__ == "__main__":
    main()
