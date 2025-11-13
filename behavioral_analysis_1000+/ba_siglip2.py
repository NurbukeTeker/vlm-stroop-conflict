import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CONFIGURATION
# ==========================================================

# Dataset structure:
# stroop_dataset_v2/
#    ├── congruent/
#    └── incongruent/
BASE_DATASET_DIR = r"..\dataset_generation\stroop_dataset_v2"

CONG_DIR   = os.path.join(BASE_DATASET_DIR, "congruent")
INCONG_DIR = os.path.join(BASE_DATASET_DIR, "incongruent")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Dataset root: {BASE_DATASET_DIR}")

# SigLIP Model
MODEL_ID = "google/siglip-base-patch16-384"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
model.eval()

# Colors (same order as dataset)
ALL_COLORS = ["black","blue","brown","gray","green","orange","pink","purple","red","yellow"]


# ==========================================================
# HELPERS
# ==========================================================

def get_image_embedding(img):
    """Convert PIL Image → SigLIP image embedding"""
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.get_image_features(**inputs)

    out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy()


def get_text_embedding(prompts):
    """Convert list of prompts → SigLIP text embeddings"""
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model.get_text_features(**inputs)

    out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy()


# ==========================================================
# LOAD IMAGES (BOTH FOLDERS)
# ==========================================================

image_paths = []

for f in os.listdir(CONG_DIR):
    if f.endswith(".png"):
        image_paths.append(os.path.join(CONG_DIR, f))

for f in os.listdir(INCONG_DIR):
    if f.endswith(".png"):
        image_paths.append(os.path.join(INCONG_DIR, f))

print(f"Total images found: {len(image_paths)}")


# ==========================================================
# PROMPT SETS
# ==========================================================

word_prompts = [f"The text says {c.upper()}" for c in ALL_COLORS]
ink_prompts  = [f"The text is written in {c} color" for c in ALL_COLORS]

word_text_embs = get_text_embedding(word_prompts)   # semantic prompts
ink_text_embs  = get_text_embedding(ink_prompts)    # perceptual prompts


# ==========================================================
# MAIN EVALUATION LOOP
# ==========================================================

records = []

for path in tqdm(image_paths, desc="Evaluating SigLIP-2"):
    fname = os.path.basename(path).lower().replace(".png","")

    if "_as_" not in fname:
        continue

    # filename structure: WORD_as_INK_bg-XXXX_Txx_Sxx_Bxx
    word_raw = fname.split("_as_")[0]
    ink_raw  = fname.split("_as_")[1].split("_")[0]

    word = word_raw.lower()
    ink  = ink_raw.lower()

    condition = "congruent" if word == ink else "incongruent"

    # LOAD image
    img = Image.open(path).convert("RGB")

    # SigLIP image embedding
    img_emb = get_image_embedding(img)

    # TEXT (word) prediction
    sim_word = cosine_similarity(img_emb, word_text_embs)[0]
    pred_word = ALL_COLORS[np.argmax(sim_word)]

    # INK (ink color) prediction
    sim_ink = cosine_similarity(img_emb, ink_text_embs)[0]
    pred_ink = ALL_COLORS[np.argmax(sim_ink)]

    records.append({
        "file": fname,
        "word": word,
        "ink": ink,
        "condition": condition,
        "pred_word": pred_word,
        "pred_ink": pred_ink,
        "word_correct": int(pred_word == word),
        "ink_correct": int(pred_ink == ink)
    })


df = pd.DataFrame(records)


# ==========================================================
# RESULTS
# ==========================================================

print("\n=== SigLIP-2 Stroop Results ===")

def acc(col, cond):
    subset = df[df["condition"] == cond]
    return subset[col].mean() if len(subset) else 0

print(f"Congruent   → Word Acc = {acc('word_correct','congruent'):.3f}, Ink Acc = {acc('ink_correct','congruent'):.3f}")
print(f"Incongruent → Word Acc = {acc('word_correct','incongruent'):.3f}, Ink Acc = {acc('ink_correct','incongruent'):.3f}")

df.to_csv("siglip2_results_v2.csv", index=False)
print("\nSaved → siglip2_results_v2.csv")
