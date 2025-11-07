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
base_dir = r"c:\Users\nurte\OneDrive\Desktop\vlm-stroop-conflict\stroop_images"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Base dir: {base_dir}")

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_colors = ["black","blue","brown","gray","green","orange","pink","purple","red","yellow"]

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def get_text_embeddings(prompts):
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**text_inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# ==========================
# LOAD ALL IMAGES
# ==========================
image_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".png"):
            image_files.append(os.path.join(root, f))

print(f"Total images found: {len(image_files)}")

# ==========================
# PROMPT DEFINITIONS
# ==========================
text_prompts = [f"The text says {c.upper()}" for c in all_colors]
color_prompts = [f"The text is written in {c} color" for c in all_colors]

text_embs = get_text_embeddings(text_prompts)
color_embs = get_text_embeddings(color_prompts)

# ==========================
# MAIN LOOP
# ==========================
records = []

for path in tqdm(image_files, desc="Evaluating Stroop stimuli"):
    fname = os.path.basename(path).replace(".png", "").lower()

    # Robust parse for "word_as_color"
    if "_as_" not in fname:
        print(f"❌ Skipping (bad name): {fname}")
        continue

    word, color = fname.split("_as_")
    condition = "congruent" if word == color else "incongruent"

    image_emb = get_embedding(path)

    # Compute similarities
    sim_text = cosine_similarity(image_emb, text_embs)[0]
    sim_color = cosine_similarity(image_emb, color_embs)[0]

    word_pred = all_colors[np.argmax(sim_text)]
    ink_pred  = all_colors[np.argmax(sim_color)]

    records.append({
        "file": fname,
        "word": word,
        "ink": color,
        "condition": condition,
        "pred_word": word_pred,
        "pred_ink": ink_pred,
        "word_correct": int(word_pred == word),
        "ink_correct": int(ink_pred == color)
    })

df = pd.DataFrame(records)

# ==========================
# RESULTS
# ==========================
print("\n✅ CLIP Stroop Prompt Similarity Results")

if len(df) == 0:
    print("⚠️ No valid data parsed. Check filename pattern.")
else:
    def acc(col, cond):
        subset = df[df["condition"] == cond]
        return subset[col].mean() if not subset.empty else 0.0

    print(f"📗 Congruent — Word acc: {acc('word_correct','congruent'):.3f}, Ink acc: {acc('ink_correct','congruent'):.3f}")
    print(f"📕 Incongruent — Word acc: {acc('word_correct','incongruent'):.3f}, Ink acc: {acc('ink_correct','incongruent'):.3f}")

    df.to_csv("clip_stroop_results.csv", index=False)
    print("\n📂 Saved results to clip_stroop_results.csv")
