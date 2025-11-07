"""
CLIP Visual Manipulation — Contrast Analysis (Full Working Version)
------------------------------------------------------------------
Evaluates CLIP model performance on Stroop-style stimuli
under different background contrast levels (both congruent & incongruent).

Expected folder structure:
    behavioural_analysis/
        visual_manipulations/
            contrast_manipulation/
                stroop_contrast_variants/
                stroop_incongruent_contrast_variants/
                results/
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ==========================
# CONFIGURATION
# ==========================
# Automatically build absolute paths (works from anywhere)
BASE_DIR = os.path.join(os.getcwd(), "behavioural_analysis", "visual_manipulations", "contrast_manipulation")
CONGRUENT_DIR = os.path.join(BASE_DIR, "stroop_contrast_variants")
INCONGRUENT_DIR = os.path.join(BASE_DIR, "stroop_incongruent_contrast_variants")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "openai/clip-vit-base-patch32"
PROMPT_TEMPLATE = "The word is written in {} font."
COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "brown", "pink", "gray", "grey", "black"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD MODEL
# ==========================
print(f"🔹 Loading CLIP model: {MODEL_ID}")
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_contrast_from_filename(fname):
    """Extracts contrast level (bg_high, bg_low, bg_medium, bg_same)."""
    name = fname.lower()
    if "bg_" in name:
        return name.split("bg_")[1].replace(".png", "")
    return "unknown"


def evaluate_stroop_set(folder_path, is_incongruent):
    """Evaluates one dataset (congruent or incongruent)."""
    condition = "incongruent" if is_incongruent else "congruent"
    results = []

    print(f"\n🧩 Reading directory: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return pd.DataFrame()

    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    print(f"Found subfolders: {subfolders}")

    for word in COLOR_WORDS:
        word_dir = os.path.join(folder_path, word)
        if not os.path.isdir(word_dir):
            continue

        files = [f for f in os.listdir(word_dir) if f.lower().endswith(".png")]
        if not files:
            print(f"⚠️ No PNG files found in {word_dir}")
            continue

        for filename in sorted(files):
            image_path = os.path.join(word_dir, filename)
            image = Image.open(image_path).convert("RGB")

            if is_incongruent:
                # Example: red_as_blue_bg_high.png
                try:
                    word_text = filename.split("_as_")[0]
                    font_color = filename.split("_as_")[1].split("_bg_")[0]
                    contrast = filename.split("_bg_")[1].replace(".png", "")
                except Exception as e:
                    print(f"⚠️ Could not parse {filename}: {e}")
                    continue
            else:
                # Example: red_bg_high.png
                try:
                    word_text = filename.split("_bg_")[0]
                    font_color = word_text
                    contrast = get_contrast_from_filename(filename)
                except Exception as e:
                    print(f"⚠️ Could not parse {filename}: {e}")
                    continue

            # Run CLIP inference
            prompts = [PROMPT_TEMPLATE.format(c) for c in COLOR_WORDS]
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0]
                probs = logits.softmax(dim=0)
                top_idx = torch.argmax(probs).item()

            predicted_color = COLOR_WORDS[top_idx]

            results.append({
                "image": filename,
                "word_text": word_text,
                "font_color": font_color,
                "predicted_color": predicted_color,
                "contrast": contrast,
                "condition": condition,
                "is_word_prediction": predicted_color == word_text,
                "is_color_prediction": predicted_color == font_color,
            })

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, f"clip_stroop_{condition}_contrast_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {condition} results to {out_csv}")
    return df


def summarize_results(df, condition):
    """Aggregates accuracy by contrast level."""
    if "contrast" not in df.columns or df.empty:
        print(f"⚠️ No valid data found for {condition} set — skipping summary.")
        return pd.DataFrame()

    summary = (
        df.groupby("contrast")
        .agg(
            total_images=("image", "count"),
            word_matches=("is_word_prediction", "sum"),
            color_matches=("is_color_prediction", "sum")
        )
        .assign(
            word_accuracy=lambda d: (d["word_matches"] / d["total_images"] * 100).round(2),
            color_accuracy=lambda d: (d["color_matches"] / d["total_images"] * 100).round(2),
            condition=condition
        )[["contrast", "word_accuracy", "color_accuracy", "word_matches", "color_matches", "total_images", "condition"]]
        .reset_index(drop=True)
    )

    out_csv = os.path.join(RESULTS_DIR, f"clip_stroop_{condition}_contrast_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"📊 Saved {condition} summary to {out_csv}")
    print(summary)
    return summary


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    # Congruent
    df_congruent = evaluate_stroop_set(CONGRUENT_DIR, is_incongruent=False)
    summary_congruent = summarize_results(df_congruent, "congruent")

    # Incongruent
    df_incongruent = evaluate_stroop_set(INCONGRUENT_DIR, is_incongruent=True)
    summary_incongruent = summarize_results(df_incongruent, "incongruent")

    # Merge
    if not summary_congruent.empty or not summary_incongruent.empty:
        combined = pd.concat([summary_congruent, summary_incongruent], ignore_index=True)
        combined_out = os.path.join(RESULTS_DIR, "clip_contrast_combined_summary.csv")
        combined.to_csv(combined_out, index=False)
        print("\n=== Combined Contrast Summary ===")
        print(combined)
        print(f"\n✅ Combined summary saved to {combined_out}")
    else:
        print("⚠️ No data available to merge — please check folder structure and file names.")


if __name__ == "__main__":
    main()


