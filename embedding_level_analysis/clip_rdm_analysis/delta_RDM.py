"""
Final CLIP RDM generator + visualization (pure blue–white–red scale)
--------------------------------------------------------------------
 - Filters congruent Stroop images (word == ink)
 - Computes RDMs and ΔRDMs
 - Generates publication-ready heatmaps using pure blue–white–red colormap
"""

import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.colors import LinearSegmentedColormap

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

DATA_PATHS = {
    "stroop": Path("stroop_images"),  # only congruent
    "grayscale": Path("embedding_level_analysis/clip_rdm_analysis/grayscale_images"),
    "color": Path("embedding_level_analysis/clip_rdm_analysis/color_patches_single"),
}

SAVE_DIR = Path("embedding_level_analysis/clip_rdm_analysis/rdm_heatmaps")
FIG_DIR = SAVE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def embed_images(folder: Path):
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png images found in {folder.resolve()}")
    embeddings = []
    for path in tqdm(paths, desc=f"Embedding {folder.name}"):
        image = preprocess(Image.open(path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(image).cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

def embed_stroop_congruent(folder: Path):
    paths = [p for p in folder.rglob("*.png") if "_as_" in p.stem and p.stem.split("_as_")[0] == p.stem.split("_as_")[1]]
    if not paths:
        raise FileNotFoundError(f"No congruent Stroop images found in {folder.resolve()}")
    embeddings = []
    for path in tqdm(paths, desc="Embedding congruent Stroop"):
        image = preprocess(Image.open(path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(image).cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

def compute_rdm(embeddings):
    """1 - cosine similarity"""
    sim = 1 - cosine_distances(embeddings)
    return 1 - sim

# ---------------- RDM GENERATION ----------------
RDMs = {}
for key, path in DATA_PATHS.items():
    if not path.exists():
        print(f"⚠️ Missing folder: {path}")
        continue
    if key == "stroop":
        embs = embed_stroop_congruent(path)
    else:
        embs = embed_images(path)
    RDMs[key] = compute_rdm(embs)
    np.save(SAVE_DIR / f"RDM_{key}.npy", RDMs[key])
    print(f"✅ Saved RDM_{key}.npy")

if all(k in RDMs for k in ["stroop", "color", "grayscale"]):
    ΔWord = RDMs["stroop"] - RDMs["color"]
    ΔInk = RDMs["stroop"] - RDMs["grayscale"]
    np.save(SAVE_DIR / "ΔWord.npy", ΔWord)
    np.save(SAVE_DIR / "ΔInk.npy", ΔInk)
    print("✅ Saved ΔWord.npy and ΔInk.npy")

# ---------------- VISUALIZATION ----------------
sns.set_context("talk")
sns.set_style("white")

labels = ["BLACK", "BLUE", "BROWN", "GRAY", "GREEN",
          "ORANGE", "PINK", "PURPLE", "RED", "YELLOW"]

vmin, vmax = 0, 0.35

# Pure blue–white–red colormap
pure_blue_red = LinearSegmentedColormap.from_list("blue_white_red",
                                                  ["#08306B", "white", "#67000D"])

def plot_rdm_pair(rdm_a, rdm_b, title_a, title_b, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, rdm, title in zip(axes, [rdm_a, rdm_b], [title_a, title_b]):
        sns.heatmap(rdm, ax=ax, cmap=pure_blue_red, square=True,
                    xticklabels=labels, yticklabels=labels,
                    vmin=vmin, vmax=vmax, cbar_kws={"label": "Dissimilarity"})
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=400)
    plt.close()

# --- Grayscale vs Stroop ---
plot_rdm_pair(RDMs["grayscale"], RDMs["stroop"],
              "(a) Grayscale Stroop (Shape Only)",
              "(b) Colorful Stroop (Color + Shape)",
              "RDM_gray_vs_stroop_blue_red.png")

# --- Color vs Stroop ---
plot_rdm_pair(RDMs["color"], RDMs["stroop"],
              "(a) Solid Color (Color Only)",
              "(b) Colorful Stroop (Color + Shape)",
              "RDM_color_vs_stroop_blue_red.png")

# --- ΔRDMs ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, Δ, title in zip(axes, [ΔWord, ΔInk],
                        ["ΔRDM: Color → Color+Shape (ΔShape)",
                         "ΔRDM: Shape → Color+Shape (ΔColor)"]):
    sns.heatmap(Δ, ax=ax, cmap=pure_blue_red, square=True,
                xticklabels=labels, yticklabels=labels,
                center=0, cbar_kws={"label": "Δ Dissimilarity"})
    ax.set_title(title, fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "Delta_RDMs_blue_red.png", dpi=400)
plt.close()

print(f"✅ All RDMs and blue–red heatmaps saved to {FIG_DIR.resolve()}")
