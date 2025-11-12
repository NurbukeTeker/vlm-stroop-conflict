import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# === PATHS ===
base_dir = os.path.dirname(__file__)
folders = {
    "shape": os.path.join(base_dir, "images_black_text"),
    "color": os.path.join(base_dir, "images_color_only"),
    "colorful": os.path.join(base_dir, "stroop_images"),
}
out_dir = os.path.join(base_dir, "figures_siglip2_full")
os.makedirs(out_dir, exist_ok=True)

# === MODEL ===
model_id = "google/siglip-base-patch16-384"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

# === HELPERS ===
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def compute_rdm(embeddings):
    return pairwise_distances(embeddings, metric="cosine")

def plot_rdm(matrix, title, save_path, labels=None, vmin=0.0, vmax=0.35):
    plt.figure(figsize=(6,6))
    im = plt.imshow(matrix, cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Dissimilarity", rotation=270, labelpad=15)
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
        plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.title(title, fontsize=10, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === STEP 1: Extract embeddings ===
embeddings = {}
for key, path in folders.items():
    embs = []
    if key == "colorful":
        for subfolder in sorted(os.listdir(path)):
            subdir = os.path.join(path, subfolder)
            if not os.path.isdir(subdir): 
                continue
            for file in sorted(os.listdir(subdir)):
                if file.lower().endswith(".png") and f"{subfolder}_as_{subfolder}" in file.lower():
                    embs.append(get_embedding(os.path.join(subdir, file)))
    else:
        for file in sorted(os.listdir(path)):
            if file.lower().endswith((".png",".jpg",".jpeg")):
                embs.append(get_embedding(os.path.join(path, file)))

    embeddings[key] = np.stack(embs)
    np.save(os.path.join(out_dir, f"embeddings_{key}.npy"), embeddings[key])
    print(f"✅ {key} embeddings saved, shape={embeddings[key].shape}")

# === STEP 2: Compute RDMs ===
rdms = {k: compute_rdm(v) for k,v in embeddings.items()}
labels = ["BLACK","BLUE","BROWN","GRAY","GREEN","ORANGE","PINK","PURPLE","RED","YELLOW"]

for name, rdm in rdms.items():
    np.save(os.path.join(out_dir, f"rdm_{name}.npy"), rdm)
    header = {
        "shape":"RDM for Grayscale Stroop Images (Shape-Only)",
        "color":"RDM for Solid Color Stroop Images (Color-Only)",
        "colorful":"RDM for Colorful Stroop Images (Color + Shape)"
    }[name]
    plot_rdm(rdm, f"{header}\n(Condition: {name})",
             os.path.join(out_dir, f"rdm_{name}.png"), labels)

# === STEP 3: Normalize all RDMs for consistent scale ===
def normalize_matrix(m):
    m_min, m_max = np.min(m), np.max(m)
    return (m - m_min) / (m_max - m_min + 1e-8)

for k in rdms:
    rdms[k] = normalize_matrix(rdms[k])

# === STEP 4: ΔRDMs ===
delta_shape = rdms["colorful"] - rdms["color"]
delta_color = rdms["colorful"] - rdms["shape"]
np.save(os.path.join(out_dir,"delta_shape.npy"), delta_shape)
np.save(os.path.join(out_dir,"delta_color.npy"), delta_color)

plot_rdm(delta_shape,
         "ΔShape = (Color + Shape) − (Color-Only)\nEffect of adding textual shape",
         os.path.join(out_dir,"delta_shape.png"), labels, vmin=-0.15, vmax=0.15)
plot_rdm(delta_color,
         "ΔColor = (Color + Shape) − (Shape-Only)\nEffect of adding color",
         os.path.join(out_dir,"delta_color.png"), labels, vmin=-0.15, vmax=0.15)

# === STEP 5: Combined Figure (6-panel layout) ===
fig, axes = plt.subplots(2, 3, figsize=(13,8))
titles = [
    "Shape-Only (Black-Text)",
    "Color-Only (Solid Patches)",
    "Color + Shape (Stroop Congruent)",
    "ΔShape = (Colorful − Color)",
    "ΔColor = (Colorful − Shape)",
    ""
]
mats = [rdms["shape"], rdms["color"], rdms["colorful"], delta_shape, delta_color, None]

for i, (ax, title) in enumerate(zip(axes.flat, titles)):
    if i < 3:
        vmin, vmax = 0.0, 0.35    # normal RDMs
    elif i < 5:
        vmin, vmax = -0.15, 0.15  # delta RDMs
    else:
        ax.axis("off")
        continue

    im = ax.imshow(mats[i], cmap="coolwarm", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("SigLIP-2 — Full RDM Comparison (Consistent Scale, Congruent Only)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(out_dir, "rdm_sixpanel_siglip2_corrected.png"), dpi=300)
plt.close()
