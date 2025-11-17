import os
import torch
from torch import nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# ============================================================
# PATHS (AUTO-RESOLVE BASED ON SCRIPT LOCATION)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "data"))
SAVE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "saved", "embeddings"))

os.makedirs(SAVE_ROOT, exist_ok=True)

print("[INFO] DATA_ROOT =", DATA_ROOT)
print("[INFO] SAVE_ROOT =", SAVE_ROOT)


# ============================================================
# LOAD MODEL (GPU-first, CPU-safe, no accelerate required)
# ============================================================
def load_qwen(model_id="Qwen/Qwen2-VL-7B-Instruct"):
    print("[INFO] Loading Qwen2-VL-7B-Instruct...")

    if torch.cuda.is_available():
        dtype = torch.float16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True
    )

    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"[INFO] Model loaded on {device}. dtype={dtype}.")
    return model, processor


# ============================================================
# DEBUG: INSPECT MODEL STRUCTURE
# ============================================================
def debug_model_structure(model):
    """Print key attributes to help identify the correct forward path."""
    print("\n[DEBUG] Model structure overview:")
    print(f"  - Model class: {model.__class__.__name__}")
    print(f"  - Has 'vision_tower': {hasattr(model, 'vision_tower')}")
    print(f"  - Has 'model': {hasattr(model, 'model')}")
    if hasattr(model, "model"):
        m = model.model
        print(f"    - model.class: {m.__class__.__name__}")
        print(f"    - model has 'visual': {hasattr(m, 'visual')}")
        print(f"    - model has 'vision_model': {hasattr(m, 'vision_model')}")
    print()


# ============================================================
# REGISTER HOOKS FOR QWEN2-VL VISION ENCODER
# ============================================================
def register_hooks(model):
    print("[INFO] Registering hooks for vision encoder blocks (auto-discover)...")

    collected = {}
    candidates = []

    # Search recursively for any module exposing a `.blocks` list
    for name, mod in model.named_modules():
        if hasattr(mod, "blocks"):
            try:
                blocks = getattr(mod, "blocks")
                _ = len(blocks)
                candidates.append((name, blocks))
            except:
                pass

    if not candidates:
        raise ValueError("[ERROR] No module with .blocks found in Qwen model.")

    # Pick the biggest stack (correct encoder)
    chosen_name, blocks = max(candidates, key=lambda x: len(x[1]))
    print(f"[INFO] Found candidate blocks at module '{chosen_name}' with {len(blocks)} layers.")

    # Register hooks on each transformer block
    for idx, block in enumerate(blocks):
        def make_hook(layer_idx):
            def hook_fn(_, __, output):
                collected[layer_idx] = output
            return hook_fn

        block.register_forward_hook(make_hook(idx))

    print(f"[INFO] Registered {len(blocks)} vision layers (from '{chosen_name}').")
    return collected


# ============================================================
# LIST IMAGE FILES
# ============================================================
def load_labeled_images(path):
    items = []
    if not os.path.isdir(path):
        print(f"[WARNING] Missing directory: {path}")
        return items

    # Recursively walk through all subdirectories to find images
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(root, f)
                # Use the immediate parent directory as label
                label = os.path.basename(root)
                items.append((label, fpath))

    return items


# ============================================================
# EXTRACT LAYERWISE CLS TOKEN FROM ONE IMAGE
# ============================================================
def extract_layerwise(model, processor, image_path, hooks_dict):
    img = Image.open(image_path).convert("RGB")

    device = next(model.parameters()).device

    # Qwen2-VL requires text="" even for vision-only forward
    inputs = processor(images=img, text="", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    hooks_dict.clear()

    with torch.no_grad():
        # Try multiple possible forward pass methods
        forward_success = False
        
        # Method 1: Direct vision_tower (if exposed)
        if hasattr(model, "vision_tower") and not forward_success:
            try:
                _ = model.vision_tower(**inputs)
                forward_success = True
            except Exception as e:
                print(f"[DEBUG] Method 1 (vision_tower) failed: {e}")
        
        # Method 2: Call model.model.visual as a function expecting pixel_values + image_grid_thw
        if hasattr(model, "model") and hasattr(model.model, "visual") and not forward_success:
            try:
                pixel_values = inputs.get("pixel_values")
                image_grid_thw = inputs.get("image_grid_thw")
                if pixel_values is not None and image_grid_thw is not None:
                    # Try as positional arguments first (common in vision transformers)
                    _ = model.model.visual(pixel_values, image_grid_thw)
                    forward_success = True
            except Exception as e:
                print(f"[DEBUG] Method 2 (visual positional) failed: {e}")
        
        # Method 3: Full model forward pass (captures vision through full pipeline)
        if not forward_success:
            try:
                _ = model(**inputs)
                forward_success = True
            except Exception as e:
                print(f"[DEBUG] Method 3 (full forward) failed: {e}")
                raise ValueError(f"[ERROR] All forward pass methods failed. Last error: {e}")
        
        if not forward_success:
            raise ValueError("[ERROR] Could not execute any forward pass method.")

    layer_embed = {}

    for layer_idx, output in hooks_dict.items():
        # Debug: Print shape on first layer to understand structure
        if layer_idx == 0:
            print(f"[DEBUG] Layer 0 output shape: {output.shape}")
        
        # Handle different output shapes
        if output.dim() == 3:
            # Shape: [batch, seq_len, hidden_dim] → extract CLS (first token)
            cls = output[:, 0, :].detach().float().cpu()
        elif output.dim() == 2:
            # Shape: [batch, hidden_dim] → already reduced, use as-is
            cls = output.detach().float().cpu()
        else:
            # Unexpected shape, print and skip
            print(f"[WARNING] Layer {layer_idx} has unexpected shape {output.shape}, skipping.")
            continue
        
        # Normalize per-vector (avoid divide by zero)
        cls = cls / (cls.norm(dim=-1, keepdim=True) + 1e-12)
        layer_embed[layer_idx] = cls

    return layer_embed


# ============================================================
# SAVE EMBEDDINGS
# ============================================================
def save_embedding(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor, save_path)


# ============================================================
# PROCESS ONE DATASET
# ============================================================
def process_dataset(model, processor, hooks, dataset_name, data_path):
    print(f"[INFO] Processing dataset → {dataset_name}")

    samples = load_labeled_images(data_path)
    print(f"[INFO] Found {len(samples)} samples in [{dataset_name}]")

    if len(samples) == 0:
        print(f"[INFO] Skipping dataset '{dataset_name}' (no samples).")
        return

    for idx, (label, img_path) in enumerate(samples):
        print(f"[INFO] [{dataset_name}] {idx+1}/{len(samples)} → {img_path}")

        layer_embeds = extract_layerwise(model, processor, img_path, hooks)

        for layer_idx, emb in layer_embeds.items():
            save_dir = os.path.join(
                SAVE_ROOT, dataset_name, label, f"layer{layer_idx}"
            )
            fname = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
            save_embedding(emb, os.path.join(save_dir, fname))

    print(f"[INFO] Completed dataset: {dataset_name}")


# ============================================================
# MAIN
# ============================================================
def main():
    model, processor = load_qwen()
    debug_model_structure(model)
    hooks = register_hooks(model)

    datasets = {
        "congruent": os.path.join(DATA_ROOT, "congruent"),
        "color_only": os.path.join(DATA_ROOT, "color_only"),
        "text_only": os.path.join(DATA_ROOT, "text_only"),
    }

    for name, path in datasets.items():
        process_dataset(model, processor, hooks, name, path)

    print("[INFO] All embeddings extracted and saved successfully.")


if __name__ == "__main__":
    main()
