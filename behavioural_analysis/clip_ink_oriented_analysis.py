"""
CLIP Behavioural Analysis — Ink-Oriented Test
Tests how well CLIP identifies the ink color (visual feature)
instead of the written word (textual feature).
"""

import os
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Base directory (alt klasörleriyle birlikte)
base_dir = os.path.join(os.path.dirname(__file__), "..", "stroop_images")
base_dir = os.path.abspath(base_dir)
print("Base dir:", base_dir)

# Renk listesi ve prompt ailesi
colors = ["black","blue","brown","gray","green","orange","pink","purple","red","yellow"]
ink_prompts = [f"The text is written in {c} color" for c in colors]

results = []

# 🔍 Alt klasörleri de tara
image_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".png"):
            image_files.append(os.path.join(root, f))
print(f"Found {len(image_files)} total .png files")

# === Main loop ===
for img_path in tqdm(sorted(image_files), desc="Ink-oriented test"):
    fname = os.path.basename(img_path)
    word, _, color = fname.replace(".png","").partition("_as_")
    word, color = word.lower(), color.lower()
    if not word or not color:
        print(f"⚠️ Skipping invalid filename: {fname}")
        continue

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        txt_tokens = clip.tokenize(ink_prompts).to(device)
        txt_feat = model.encode_text(txt_tokens)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).softmax(dim=-1)
    pred_ink = colors[sims.argmax(dim=-1).item()]

    results.append({
        "file": fname,
        "word_color": word,
        "true_ink": color,
        "pred_ink": pred_ink,
        "correct": int(pred_ink == color),
        "condition": "congruent" if word == color else "incongruent"
    })

# === Save results ===
df = pd.DataFrame(results)
os.makedirs("behavioural_analysis/results", exist_ok=True)
out_path = "behavioural_analysis/results/clip_ink_oriented_openai.csv"
df.to_csv(out_path, index=False)

# === Summary ===
print(f"\n✅ CLIP Ink-Oriented Test Done ({len(df)} images)")
if len(df) > 0:
    print("🎨 Congruent acc:", df[df.condition=='congruent'].correct.mean())
    print("🧩 Incongruent acc:", df[df.condition=='incongruent'].correct.mean())
else:
    print("⚠️ No valid images processed. Check filename pattern or folder depth.")
