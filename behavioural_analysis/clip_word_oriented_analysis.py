import os
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_dir = os.path.join(os.path.dirname(__file__), "..", "stroop_images")
base_dir = os.path.abspath(base_dir)
print("Base dir:", base_dir)

colors = ["black","blue","brown","gray","green","orange","pink","purple","red","yellow"]
word_prompts = [f"The text says {c.upper()}" for c in colors]

results = []

# 🔍 Walk through all subfolders
image_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".png"):
            image_files.append(os.path.join(root, f))
print(f"Found {len(image_files)} total .png files")

for img_path in tqdm(sorted(image_files), desc="Word-oriented test"):
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

        txt_tokens = clip.tokenize(word_prompts).to(device)
        txt_feat = model.encode_text(txt_tokens)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).softmax(dim=-1)
    pred_word = colors[sims.argmax(dim=-1).item()]

    results.append({
        "file": fname,
        "true_word": word,
        "ink_color": color,
        "pred_word": pred_word,
        "correct": int(pred_word == word),
        "condition": "congruent" if word == color else "incongruent"
    })

df = pd.DataFrame(results)
os.makedirs("behavioural_analysis/results", exist_ok=True)
out_path = "behavioural_analysis/results/clip_word_oriented_openai.csv"
df.to_csv(out_path, index=False)

print(f"\n✅ CLIP Word-Oriented Test Done ({len(df)} images)")
if len(df) > 0:
    print("📘 Congruent acc:", df[df.condition=='congruent'].correct.mean())
    print("📕 Incongruent acc:", df[df.condition=='incongruent'].correct.mean())
else:
    print("⚠️ No valid images processed. Check filename pattern or folder depth.")
