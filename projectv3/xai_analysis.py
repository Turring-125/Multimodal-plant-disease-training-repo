"""
xai_analysis.py — v3

Explainable AI (XAI) for the multimodal crop disease model.

TWO explanation streams:
  1. IMAGE  — Grad-CAM heatmap on EfficientNetV2-S's last conv layer
  2. WEATHER — Attention weight heatmap + per-timestep importance bar chart

Output per sample:
  A single figure with 3 panels:
    [Original Image] | [Grad-CAM Overlay] | [Weather Attention Heatmap]
  Saved as: xai_outputs/xai_{sample_id}_{crop}_{disease}.png

Usage:
  python xai_analysis.py --n_samples 10 --variant multimodal
  python xai_analysis.py --n_samples 5 --variant image-only
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECTV3_PATH = r"d:\clg\major project\Project folder\projectv3"
DATASET_DIR    = os.path.join(PROJECTV3_PATH, "dataset")
INDEX_PATH     = os.path.join(DATASET_DIR, "dataset_index.json")
WEATHER_DIR    = os.path.join(DATASET_DIR, "weather")
STATS_PATH     = os.path.join(DATASET_DIR, "weather_stats.npy")
OUTPUT_DIR     = os.path.join(PROJECTV3_PATH, "outputs", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Import model definitions from train_multimodal ────────────────────────────

from train_multimodal import (
    MultimodalDiseaseNet, DISEASE_MAPS, CLASS_LIST, LABEL_MAP,
    WINDOW_DAYS, N_FEATURES, IMG_SIZE
)

# ── Feature names for weather axis labels ─────────────────────────────────────

FEATURE_NAMES = [
    "Temp Max (°C)", "Temp Min (°C)", "Temp Mean (°C)",
    "RH Max (%)", "RH Mean (%)", "Precip (mm)",
    "Dewpoint (°C)", "Wind Max (km/h)", "Soil Temp (°C)",
    "Month Sin", "Month Cos",
]

# ── Image preprocessing ───────────────────────────────────────────────────────

def get_img_transform():
    return T.Compose([
        T.Resize(int(IMG_SIZE * 1.1)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def denormalize_image(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE XAI — Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM implementation for EfficientNetV2-S in timm.

    Target layer: model.image_encoder.conv_head
    (last conv layer before global average pooling)
    """

    def __init__(self, model, target_layer_name="conv_head"):
        self.model = model
        self._gradients = None
        self._activations = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, target_layer_name):
        """Register forward and backward hooks on the target layer."""
        target_layer = None

        # Try to find the layer in image_encoder
        if hasattr(self.model, "image_encoder"):
            backbone = self.model.image_encoder
            if hasattr(backbone, target_layer_name):
                target_layer = getattr(backbone, target_layer_name)

        if target_layer is None:
            # Fallback: search recursively
            for name, module in self.model.named_modules():
                if target_layer_name in name and isinstance(module, nn.Conv2d):
                    target_layer = module
                    break

        if target_layer is None:
            raise ValueError(
                f"Could not find layer '{target_layer_name}' in model. "
                "Check model.named_modules() output."
            )

        self._hook_handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hook_handles.append(
            target_layer.register_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=None):
        """
        Generates a Grad-CAM heatmap for the given image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Shape [1, 3, H, W], already on the correct device.
        target_class : int or None
            Class index to explain. If None, uses argmax of model output.

        Returns
        -------
        heatmap : np.ndarray
            Shape [H, W], values in [0, 1]. Ready to overlay on image.
        predicted_class : int
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        outputs = self.model(image_tensor,
                              torch.zeros(1, WINDOW_DAYS, N_FEATURES,
                                          device=image_tensor.device))
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Backward pass — gradient w.r.t. target class score
        self.model.zero_grad()
        outputs[0, target_class].backward()

        # Pool gradients over spatial dimensions
        gradients   = self._gradients   # [1, C, H', W']
        activations = self._activations  # [1, C, H', W']

        weights  = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam      = (weights * activations).sum(dim=1)         # [1, H', W']
        cam      = torch.relu(cam).squeeze(0)                  # [H', W']

        # Normalize and upsample to input size
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_np = cam.cpu().numpy()

        # Resize to match original image
        h, w   = image_tensor.shape[2], image_tensor.shape[3]
        cam_np = cv2.resize(cam_np, (w, h))

        return cam_np, target_class

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()


def overlay_gradcam(original_img_np, heatmap, alpha=0.45):
    """
    Overlays a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img_np : np.ndarray  [H, W, 3] float32 in [0, 1]
    heatmap : np.ndarray          [H, W] float32 in [0, 1]
    alpha : float                 Blend factor for heatmap

    Returns
    -------
    np.ndarray [H, W, 3] float32 in [0, 1]
    """
    colormap  = cm.get_cmap("jet")
    heatmap_c = colormap(heatmap)[:, :, :3]  # [H, W, 3]
    overlay   = (1 - alpha) * original_img_np + alpha * heatmap_c
    return np.clip(overlay, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER XAI — Attention weights
# ══════════════════════════════════════════════════════════════════════════════

def get_weather_attention(model, weather_tensor):
    """
    Runs a forward pass on the weather encoder and returns attention weights.

    Parameters
    ----------
    model : MultimodalDiseaseNet
    weather_tensor : torch.Tensor [1, 7, 11]

    Returns
    -------
    attn_weights : np.ndarray [7, 7]
        Averaged across attention heads.
    """
    model.eval()
    with torch.no_grad():
        if model.variant == "multimodal":
            # Dummy image to satisfy forward signature
            dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                                    device=weather_tensor.device)
            _, attn_w = model(dummy_img, weather_tensor, return_attention=True)
        elif model.variant == "weather-only":
            _, attn_w = model(None, weather_tensor, return_attention=True)
        else:
            return None

    return attn_w.squeeze(0).cpu().numpy()  # [7, 7]


def plot_weather_attention(attn_weights, wx_normalized, ax_heatmap, ax_bar,
                            title_suffix=""):
    """
    Plots the attention weight heatmap and per-timestep importance bar chart.

    Parameters
    ----------
    attn_weights : np.ndarray [7, 7]
    wx_normalized : np.ndarray [7, 11]  (z-score normalized)
    ax_heatmap : matplotlib Axes
    ax_bar : matplotlib Axes
    title_suffix : str
    """
    day_labels = [f"Day {i+1}" for i in range(WINDOW_DAYS)]

    # Heatmap: query × key attention
    im = ax_heatmap.imshow(attn_weights, cmap="YlOrRd", aspect="auto",
                            vmin=0, vmax=attn_weights.max())
    ax_heatmap.set_xticks(range(WINDOW_DAYS))
    ax_heatmap.set_yticks(range(WINDOW_DAYS))
    ax_heatmap.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax_heatmap.set_yticklabels(day_labels, fontsize=8)
    ax_heatmap.set_xlabel("Key (source day)", fontsize=8)
    ax_heatmap.set_ylabel("Query (attending day)", fontsize=8)
    ax_heatmap.set_title(f"Attention Weights{title_suffix}", fontsize=9,
                          fontweight="bold")
    plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    # Bar chart: which day is most globally attended (column-sum = how much
    # other days attend to day i)
    importance = attn_weights.sum(axis=0)  # [7] — how attended each day is
    importance = importance / importance.sum()

    colors = plt.cm.YlOrRd(importance / importance.max())
    bars = ax_bar.bar(day_labels, importance, color=colors, edgecolor="black",
                       linewidth=0.5)
    ax_bar.set_xlabel("Day", fontsize=8)
    ax_bar.set_ylabel("Attention Score (normalized)", fontsize=8)
    ax_bar.set_title("Per-Day Importance", fontsize=9, fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=45, labelsize=8)

    # Annotate bars
    for bar, val in zip(bars, importance):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
#  Composite XAI figure
# ══════════════════════════════════════════════════════════════════════════════

def generate_xai_figure(model, record, device, stats, grad_cam, transform,
                          output_dir):
    """
    Generates and saves an XAI figure for a single sample.

    Layout (1 row, 4 columns):
      [Original Image] | [Grad-CAM] | [Attention Heatmap] | [Day Importance]

    Parameters
    ----------
    model : MultimodalDiseaseNet
    record : dict  (one entry from dataset_index.json)
    device : torch.device
    stats : dict  {'mean': np.ndarray [11], 'std': np.ndarray [11]}
    grad_cam : GradCAM or None
    transform : torchvision transform
    output_dir : str
    """
    sample_id = record["id"]
    crop      = record["crop_type"]
    disease   = record["disease_label"]

    # ── Load and preprocess image ──────────────────────────────────────────
    try:
        pil_img = Image.open(record["image_path"]).convert("RGB")
    except Exception as e:
        print(f"  [SKIP] Cannot open image {record['image_path']}: {e}")
        return

    img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, H, W]
    original_np = denormalize_image(img_tensor.squeeze(0))    # [H, W, 3]

    # ── Load and normalize weather ──────────────────────────────────────────
    wx_path = os.path.join(WEATHER_DIR,
                            record.get("weather_file", f"{sample_id}_weather.npy"))
    if not os.path.exists(wx_path):
        print(f"  [SKIP] Weather file not found: {wx_path}")
        return

    wx_raw = np.load(wx_path).astype(np.float32)          # [7, 11]
    wx_norm = (wx_raw - stats["mean"]) / (stats["std"] + 1e-8)
    wx_tensor = torch.tensor(wx_norm).unsqueeze(0).to(device)  # [1, 7, 11]

    # ── Forward pass for prediction ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor, wx_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        pred_class = outputs.argmax(dim=1).item()

    pred_name = CLASS_LIST[pred_class]
    true_name = f"{crop}_{disease}"
    true_class = LABEL_MAP.get(true_name, -1)
    correct = (pred_class == true_class)

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    heatmap = None
    if grad_cam is not None:
        try:
            heatmap, _ = grad_cam.generate(img_tensor, target_class=true_class)
        except Exception as e:
            print(f"  [WARN] Grad-CAM failed for {sample_id}: {e}")

    # ── Weather attention ──────────────────────────────────────────────────
    attn_weights = None
    if model.variant in ("multimodal", "weather-only"):
        try:
            attn_weights = get_weather_attention(model, wx_tensor)  # [7, 7]
        except Exception as e:
            print(f"  [WARN] Attention extraction failed: {e}")

    # ── Build figure ───────────────────────────────────────────────────────
    has_gradcam = heatmap is not None
    has_attn    = attn_weights is not None

    n_panels = 1 + int(has_gradcam) + int(has_attn) * 2
    figw = 4.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(figw, 5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # Panel 1: Original image
    axes[ax_idx].imshow(original_np)
    axes[ax_idx].axis("off")
    status = "✓ CORRECT" if correct else "✗ WRONG"
    axes[ax_idx].set_title(
        f"Original\nTrue: {disease}\nPred: {pred_name.split('_', 1)[-1]}"
        f"  [{status}]",
        fontsize=8, fontweight="bold",
        color="green" if correct else "red"
    )
    ax_idx += 1

    # Panel 2: Grad-CAM overlay
    if has_gradcam:
        overlay = overlay_gradcam(original_np, heatmap)
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title("Grad-CAM\n(Disease Region)", fontsize=8,
                                fontweight="bold")
        ax_idx += 1

    # Panels 3 & 4: Attention heatmap + bar chart
    if has_attn:
        plot_weather_attention(
            attn_weights, wx_norm,
            axes[ax_idx], axes[ax_idx + 1],
            title_suffix=f"\n({crop} | {disease})"
        )
        ax_idx += 2

    # Super-title
    fig.suptitle(
        f"XAI Analysis — Sample {sample_id} | {crop.upper()}",
        fontsize=10, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    safe_disease = disease.replace(" ", "_").replace("/", "-")
    out_path = os.path.join(output_dir,
                             f"xai_{sample_id}_{crop}_{safe_disease}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="XAI analysis for crop disease model v3")
    parser.add_argument("--variant",    default="multimodal",
                        choices=["multimodal", "image-only", "weather-only"])
    parser.add_argument("--n_samples",  type=int, default=10,
                        help="Number of random test samples to analyse")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pth checkpoint. Defaults to best_model_{variant}.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = args.model_path or os.path.join(
        PROJECTV3_PATH, f"best_model_{args.variant}.pth"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run train_multimodal.py first."
        )

    model = MultimodalDiseaseNet(
        num_classes=len(LABEL_MAP), variant=args.variant
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    # ── Load dataset index ─────────────────────────────────────────────────
    with open(INDEX_PATH) as f:
        all_records = json.load(f)

    test_records = [r for r in all_records if r["split"] == "test"]
    if not test_records:
        print("[WARN] No test records found. Using val records instead.")
        test_records = [r for r in all_records if r["split"] == "val"]

    if not test_records:
        raise RuntimeError("No test/val records in dataset_index.json.")

    # Sample N records, stratified by crop
    n = args.n_samples
    by_crop = {}
    for r in test_records:
        by_crop.setdefault(r["crop_type"], []).append(r)

    selected = []
    n_per_crop = max(1, n // len(by_crop))
    for crop, records in by_crop.items():
        selected.extend(random.sample(records, min(n_per_crop, len(records))))

    # Fill up to n if needed
    remaining = [r for r in test_records if r not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[:max(0, n - len(selected))])
    selected = selected[:n]

    print(f"Selected {len(selected)} samples for XAI analysis")

    # ── Load weather stats ─────────────────────────────────────────────────
    stats = np.load(STATS_PATH, allow_pickle=True).item()

    # ── Set up Grad-CAM ────────────────────────────────────────────────────
    grad_cam = None
    if args.variant in ("multimodal", "image-only"):
        try:
            grad_cam = GradCAM(model, target_layer_name="conv_head")
            print("Grad-CAM initialized on 'conv_head' layer.")
        except ValueError as e:
            print(f"[WARN] Grad-CAM setup failed: {e}")
            print("       Image XAI will be skipped.")

    transform = get_img_transform()

    # ── Generate XAI figures ───────────────────────────────────────────────
    print(f"\nGenerating XAI for {len(selected)} samples...")
    saved_paths = []

    for record in tqdm(selected, desc="XAI"):
        out = generate_xai_figure(
            model=model,
            record=record,
            device=device,
            stats=stats,
            grad_cam=grad_cam,
            transform=transform,
            output_dir=OUTPUT_DIR,
        )
        if out:
            saved_paths.append(out)

    if grad_cam:
        grad_cam.remove_hooks()

    print(f"\nXAI complete. {len(saved_paths)} figures saved to: {OUTPUT_DIR}")
    for p in saved_paths:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
