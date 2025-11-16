import os
import torch
from torch import nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# ============================================================
# PATHS
# ============================================================
DATA_ROOT = "data"
SAVE_ROOT = "saved/embeddings"

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================
def load_qwen(model_id="Qwen/Qwen2-VL-7B-Instruct"):
    print("[INFO] Loading Qwen2-VL-7B-Instruct...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("[INFO] Model loaded.")
    return model, processor


# ============================================================
# REGISTER HOOKS FOR LAYER-WISE FEATURE EXTRACTION
# ============================================================
def register_hooks(model):
    print("[INFO] Registering hooks for Qwen vision tower...")

    collected = {}

    for name, module in model.vision_tower.named_modules():
        if "blocks." in name and isinstance(module, nn.Module):
            layer_idx = int(name.split(".")[1])

            def make_hook(idx):
                def hook_fn(module, inp, out):
                    collected[idx] = out
                return hook_fn

            module.register_forward_hook(make_hook(layer_idx))

    print("[INFO] Hooks registered.")
    return collected


# ============================================================
# LOAD ALL IMAGES FROM A FOLDER STRUCTURE
# Each subfolder is a class (color or word)
# ============================================================
def load_labeled_images(path):
    items = []
    for label in sorted(os.listdir(path)):
        subdir = os.path.join(path, label)
        if not os.path.isdir(subdir):
            continue

        for f in os.listdir(subdir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((label, os.path.join(subdir, f)))

    return items


# ============================================================
# PROCESS ONE IMAGE → EXTRACT LAYERWISE CLS TOKENS
# ============================================================
def extract_layerwise(model, processor, image_path, hooks_dict):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(model.device)

    hooks_dict.clear()
    _ = model.vision_tower(**inputs)

    layer_embed = {}

    for layer_idx, output in hooks_dict.items():
        cls = output[:, 0, :].detach().float().cpu()
        cls = cls / cls.norm()
        layer_embed[layer_idx] = cls

    return layer_embed


# ============================================================
# SAVE EMBEDDINGS
# ============================================================
def save_embedding(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor, save_path)


# ============================================================
# MAIN EXTRACTION PROCEDURE
# ============================================================
def process_dataset(model, processor, hooks, dataset_name, data_path):
    print(f"[INFO] Processing dataset: {dataset_name}")

    samples = load_labeled_images(data_path)
    print(f"[INFO] Found {len(samples)} samples in {dataset_name}")

    for idx, (label, img_path) in enumerate(samples):
        print(f"[INFO] [{dataset_name}] {idx+1}/{len(samples)} → {img_path}")

        layer_embeds = extract_layerwise(model, processor, img_path, hooks)

        for layer_idx, emb in layer_embeds.items():
            save_path = os.path.join(
                SAVE_ROOT,
                dataset_name,
                label,
                f"layer{layer_idx}",
            )
            fname = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
            save_embedding(emb, os.path.join(save_path, fname))

    print(f"[INFO] Completed dataset: {dataset_name}")


# ============================================================
# RUN EVERYTHING
# ============================================================
def main():
    model, processor = load_qwen()
    hooks = register_hooks(model)

    datasets = {
        "congruent": os.path.join(DATA_ROOT, "congruent"),
        "color_only": os.path.join(DATA_ROOT, "color_only"),
        "text_only": os.path.join(DATA_ROOT, "text_only"),
    }

    for name, path in datasets.items():
        process_dataset(model, processor, hooks, name, path)

    print("[INFO] All embeddings extracted and saved.")


if __name__ == "__main__":
    main()
