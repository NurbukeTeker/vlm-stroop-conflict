import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from llava.model.builder import load_pretrained_model

# ============================================
# DEVICE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# PATHS
# ============================================
project_root = "C:/Github Projects/vlm-stroop-conflict/embedding_level_analysis"

folders = {
    "shape": os.path.join(project_root, "images_black_text"),
    "color": os.path.join(project_root, "images_color_only"),
    "colorful": os.path.join(project_root, "stroop_images"),
}

out_dir = os.path.join(project_root, "llava_rdm_analysis", "figures_llava")
os.makedirs(out_dir, exist_ok=True)

# ============================================
# LOAD LLaVA MODEL
# ============================================
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_path
)
model.eval()

# fp16 fix on CPU
if device == "cpu":
    model = model.float()

print(f"✅ LLaVA model loaded on {device}")

# ============================================
# HELPERS
# ============================================
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

    # fix dtype for CPU/GPU
    if device == "cpu":
        pixel_values = pixel_values.float()
    else:
        pixel_values = pixel_values.half()

    pixel_values = pixel_values.to(model.device)

    # ensure correct shape [1,3,H,W]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values.squeeze(0)

    with torch.no_grad():
        emb = model.encode_images(pixel_values)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy().flatten()


def compute_rdm(emb):
    return pairwise_distances(emb, metric="cosine")


# ============================================
# UPDATED PLOTTER (WITH COLOR LABELS)
# ============================================
def plot_rdm_shared(mat, title, save_path, vmin, vmax):
    labels = ["BLACK","BLUE","BROWN","GRAY","GREEN",
              "ORANGE","PINK","PURPLE","RED","YELLOW"]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(mat, cmap="coolwarm", interpolation="nearest",
                    vmin=vmin, vmax=vmax)

    # ticks around matrix
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(labels)), labels, fontsize=9)

    # colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Dissimilarity", rotation=270, labelpad=15)

    plt.title(title, fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================
# STEP 1 — EXTRACT EMBEDDINGS
# ============================================
embeddings = {}

for key, path in folders.items():
    embs = []
    if key == "colorful":
        # only congruent Stroop images
        for subfolder in sorted(os.listdir(path)):
            subdir = os.path.join(path, subfolder)
            if not os.path.isdir(subdir):
                continue

            for file in sorted(os.listdir(subdir)):
                name = file.lower()
                if name.endswith(".png") and f"{subfolder.lower()}_as_{subfolder.lower()}" in name:
                    embs.append(get_embedding(os.path.join(subdir, file)))
    else:
        for file in sorted(os.listdir(path)):
            if file.lower().endswith(".png"):
                embs.append(get_embedding(os.path.join(path, file)))

    embeddings[key] = np.stack(embs)
    np.save(os.path.join(out_dir, f"embeddings_{key}.npy"), embeddings[key])
    print(f"✅ embeddings_{key}.npy saved — shape={embeddings[key].shape}")

# ============================================
# STEP 2 — RDM COMPUTATION
# ============================================
rdms = {k: compute_rdm(v) for k, v in embeddings.items()}

for name, rdm in rdms.items():
    np.save(os.path.join(out_dir, f"rdm_{name}.npy"), rdm)

print("✅ RDMs computed.")

# ============================================
# STEP 3 — NORMALIZE FOR Δ
# ============================================
def norm(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-8)

for k in rdms:
    rdms[k] = norm(rdms[k])

# ============================================
# STEP 4 — ΔRDMs
# ============================================
delta_color = rdms["colorful"] - rdms["shape"]
delta_shape = rdms["colorful"] - rdms["color"]

np.save(os.path.join(out_dir, "delta_color.npy"), delta_color)
np.save(os.path.join(out_dir, "delta_shape.npy"), delta_shape)

print("✅ Delta RDMs computed.")

# ============================================
# STEP 5 — SHARED SCALE
# ============================================
try:
    clip_rdm = np.load(os.path.join(project_root, "clip_rdm_analysis", "figures_clip", "rdm_colorful.npy"))
    siglip_rdm = np.load(os.path.join(project_root, "siglip2_rdm_analysis", "figures_siglip2", "rdm_colorful.npy"))

    all_rdms = [clip_rdm, siglip_rdm, rdms["colorful"]]
    vmin, vmax = min(m.min() for m in all_rdms), max(m.max() for m in all_rdms)

    clip_dc = np.load(os.path.join(project_root, "clip_rdm_analysis", "figures_clip", "delta_color.npy"))
    clip_ds = np.load(os.path.join(project_root, "clip_rdm_analysis", "figures_clip", "delta_shape.npy"))
    sig_dc = np.load(os.path.join(project_root, "siglip2_rdm_analysis", "figures_siglip2", "delta_color.npy"))
    sig_ds = np.load(os.path.join(project_root, "siglip2_rdm_analysis", "figures_siglip2", "delta_shape.npy"))

    all_deltas = [clip_dc, clip_ds, sig_dc, sig_ds, delta_color, delta_shape]
    dvmin, dvmax = min(m.min() for m in all_deltas), max(m.max() for m in all_deltas)

    print("📏 Shared global scales computed using CLIP + SigLIP2 + LLaVA")

except:
    print("⚠️ CLIP/SigLIP2 not found. Using LLaVA-only scale.")
    vmin, vmax = rdms["colorful"].min(), rdms["colorful"].max()
    dvmin, dvmax = delta_color.min(), delta_color.max()

# ============================================
# STEP 6 — SAVE FIGURES (WITH LABELS)
# ============================================
plot_rdm_shared(rdms["shape"], "LLaVA RDM — Shape", os.path.join(out_dir, "rdm_shape_shared.png"), vmin, vmax)
plot_rdm_shared(rdms["color"], "LLaVA RDM — Color", os.path.join(out_dir, "rdm_color_shared.png"), vmin, vmax)
plot_rdm_shared(rdms["colorful"], "LLaVA RDM — Colorful", os.path.join(out_dir, "rdm_colorful_shared.png"), vmin, vmax)

plot_rdm_shared(delta_shape, "ΔShape", os.path.join(out_dir, "delta_shape_shared.png"), dvmin, dvmax)
plot_rdm_shared(delta_color, "ΔColor", os.path.join(out_dir, "delta_color_shared.png"), dvmin, dvmax)

print("\n🎉 ALL DONE! Shared-scale LLaVA RDMs saved →", out_dir)
