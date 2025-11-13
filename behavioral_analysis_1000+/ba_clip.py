import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIGURATION
# ==========================

# Dataset root
BASE_DATASET_DIR = r"..\dataset_generation\stroop_dataset_v2"

CONG_DIR  = os.path.join(BASE_DATASET_DIR, "congruent")
INCONG_DIR = os.path.join(BASE_DATASET_DIR, "incongruent")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dataset root: {BASE_DATASET_DIR}")
print(f"Device: {device}")

# Load CLIP
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# All 10 colors
ALL_COLORS = ["black","blue","brown","gray","green","orange","pink","purple","red","yellow"]


# ==========================
# HELPERS
# ==========================

def get_image_embedding(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def get_text_embedding(prompts):
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**text_inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


# ==========================
# LOAD ALL IMAGES
# ==========================

image_paths = []

# Read congruent images
for f in os.listdir(CONG_DIR):
    if f.endswith(".png"):
        image_paths.append(os.path.join(CONG_DIR, f))

# Read incongruent images
for f in os.listdir(INCONG_DIR):
    if f.endswith(".png"):
        image_paths.append(os.path.join(INCONG_DIR, f))

print(f"Total images found: {len(image_paths)}")


# ==========================
# PROMPTS
# ==========================

word_prompts = [f"The text says {c.upper()}" for c in ALL_COLORS]
ink_prompts  = [f"The text is written in {c} color" for c in ALL_COLORS]

word_text_emb = get_text_embedding(word_prompts)
ink_text_emb  = get_text_embedding(ink_prompts)


# ==========================
# MAIN LOOP
# ==========================

records = []

for path in tqdm(image_paths, desc="Evaluating CLIP"):
    fname = os.path.basename(path).lower().replace(".png","")

    if "_as_" not in fname:
        continue

    # parse filename: WORD_as_INK (rest ignored)
    word_raw = fname.split("_as_")[0]
    ink_raw  = fname.split("_as_")[1].split("_")[0]  # SAFE PARSE

    word = word_raw.lower()
    ink  = ink_raw.lower()

    condition = "congruent" if word == ink else "incongruent"

    # get image embedding
    img_emb = get_image_embedding(path)

    # word-prompt (semantic reading)
    sim_word = cosine_similarity(img_emb, word_text_emb)[0]
    pred_word = ALL_COLORS[np.argmax(sim_word)]

    # ink-prompt (ink color recognition)
    sim_ink  = cosine_similarity(img_emb, ink_text_emb)[0]
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


# ==========================
# RESULTS
# ==========================

print("\n=== CLIP Stroop Results ===")

def acc(col, cond):
    sub = df[df["condition"] == cond]
    return sub[col].mean() if len(sub) > 0 else 0

print(f"Congruent   → Word Acc = {acc('word_correct','congruent'):.3f}, Ink Acc = {acc('ink_correct','congruent'):.3f}")
print(f"Incongruent → Word Acc = {acc('word_correct','incongruent'):.3f}, Ink Acc = {acc('ink_correct','incongruent'):.3f}")

df.to_csv("clip_results_v2.csv", index=False)
print("Saved → clip_results_v2.csv")
