"""
CLIP Stroop-style Embedding Analysis (Final Version)
----------------------------------------------------
Analyzes whether CLIP’s image embeddings are closer to
the *written word* or the *ink color* in Stroop-style stimuli.
"""

import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

# --- Canonical color list ---
CANON_COLORS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "gray", "black"
]

# --- Synonyms mapping (for filename parsing) ---
SYNONYMS = {
    "grey": "gray",
    "darkgrey": "gray", "darkgray": "gray",
    "lightgrey": "gray", "lightgray": "gray",
    "maroon": "red", "scarlet": "red", "crimson": "red",
    "violet": "purple", "magenta": "purple",
    "lime": "green", "tan": "brown",
}


def canon_color(token):
    """Convert color token to canonical color if known."""
    if token is None:
        return None
    t = token.strip().lower()
    if t in CANON_COLORS:
        return t
    return SYNONYMS.get(t, None)


def parse_word_ink_from_name(path: Path):
    """
    Extract (word, ink) labels from filename patterns:
      red_as_blue.png
      word_red_ink_blue.png
      red-blue.png
    or infer from folder (e.g., red/red_as_blue.png)
    """
    name = path.stem.lower()

    # pattern 1: red_as_blue
    m = re.search(r"([a-z]+)_as_([a-z]+)", name)
    if m:
        return canon_color(m.group(1)), canon_color(m.group(2))

    # pattern 2: word_red_ink_blue
    m = re.search(r"word_([a-z]+).*ink_([a-z]+)", name)
    if m:
        return canon_color(m.group(1)), canon_color(m.group(2))

    # pattern 3: red-blue
    m = re.search(r"^([a-z]+)[\-\_]+([a-z]+)$", name)
    if m:
        return canon_color(m.group(1)), canon_color(m.group(2))

    # fallback: infer from folder name
    parent_hint = canon_color(path.parent.name)
    toks = re.findall(r"[a-z]+", name)
    guess_ink = next((canon_color(t) for t in toks if canon_color(t)), None)
    if parent_hint and guess_ink:
        return parent_hint, guess_ink

    return None, None


def get_all_images(data_dir):
    """Recursively collect all image files."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts])


def load_clip(model_id="openai/clip-vit-base-patch16"):
    """Load CLIP model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, device


def get_image_embedding(model, processor, device, image_path):
    """Get normalized CLIP image embedding."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0)


def get_text_embeddings(model, processor, device, texts):
    """Get normalized CLIP text embeddings for all colors."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb


def main():
    # --- CONFIG ---
    data_dir = "stroop_images"         # path to your Stroop dataset
    out_csv = "results/clip_embedding_results.csv"
    model_id = "openai/clip-vit-base-patch16"

    # --- LOAD MODEL ---
    model, processor, device = load_clip(model_id)
    print(f"✅ CLIP model loaded on {device}")

    # --- PRECOMPUTE TEXT EMBEDDINGS ---
    color_texts = CANON_COLORS
    text_embs = get_text_embeddings(model, processor, device, color_texts)
    print("Color text embeddings ready:", color_texts)

    # --- PROCESS IMAGES ---
    all_imgs = get_all_images(data_dir)
    print(f"Found {len(all_imgs)} images.\n")

    results = []

    for p in tqdm(all_imgs, desc="Analyzing images"):
        word, ink = parse_word_ink_from_name(p)
        try:
            img_emb = get_image_embedding(model, processor, device, p)
        except Exception as e:
            print(f"⚠️ Could not open {p}: {e}")
            continue

        sims = (img_emb @ text_embs.T).cpu().numpy()  # cosine similarities
        sims_dict = {c: float(s) for c, s in zip(color_texts, sims)}
        top_idx = int(np.argmax(sims))
        top_color = color_texts[top_idx]
        top_val = float(sims[top_idx])

        sim_word = sims_dict.get(word, None)
        sim_ink = sims_dict.get(ink, None)

        condition = None
        bias = "other"
        if word and ink:
            condition = "congruent" if word == ink else "incongruent"
            if top_color == word:
                bias = "word"
            elif top_color == ink:
                bias = "ink"

        results.append({
            "image_path": str(p),
            "word": word,
            "ink": ink,
            "condition": condition,
            "predicted": top_color,
            "cosine_word": sim_word,
            "cosine_ink": sim_ink,
            "cosine_top": top_val,
            "bias": bias,
            **{f"sim_{c}": sims_dict[c] for c in color_texts}
        })

    # --- SAVE RESULTS ---
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Results saved to: {out_csv}")

    # --- SUMMARY ---
    print("\n==== Embedding-level Stroop Summary ====")

    total = len(df)
    word_bias = (df["bias"] == "word").sum() / total * 100
    ink_bias = (df["bias"] == "ink").sum() / total * 100
    other_bias = (df["bias"] == "other").sum() / total * 100
    print(f"Overall bias: Word {word_bias:.1f}%, Ink {ink_bias:.1f}%, Other {other_bias:.1f}%")

    # Congruent vs Incongruent
    df_valid = df.dropna(subset=["condition"])
    if not df_valid.empty:
        summary = (
            df_valid.groupby("condition")["bias"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0) * 100
        )
        print("\nBias by condition (%):")
        print(summary.round(1))

    # Average cosine similarities
    avg_word_sim = df["cosine_word"].dropna().mean()
    avg_ink_sim = df["cosine_ink"].dropna().mean()
    print(f"\nAverage cosine similarity → Word: {avg_word_sim:.4f} | Ink: {avg_ink_sim:.4f}")
    if avg_word_sim > avg_ink_sim:
        print("➡️ CLIP embeddings are overall closer to the WORD representation.")
    else:
        print("➡️ CLIP embeddings are overall closer to the INK representation.")


if __name__ == "__main__":
    main()
