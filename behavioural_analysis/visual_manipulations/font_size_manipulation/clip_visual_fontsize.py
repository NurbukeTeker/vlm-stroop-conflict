"""
CLIP Visual Manipulation — Font Size Analysis
---------------------------------------------
Runs CLIP on Stroop-style datasets with varying font sizes
(48pt, 72pt, 90pt, 108pt), evaluates text vs. color bias,
and saves results and summary CSVs.

Folder structure expected:
    behavioural_analysis/
        visual_manipulations/
            clip_visual_fontsize.py
    stroop_images_base_48/
    stroop_images_base_72/
    stroop_images_base_90/
    stroop_images_base_108/
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ==========================
# CONFIGURATION
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

FONT_SETS = {
    "48pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_48",
    "72pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_72",
    "90pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_90",
    "108pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_108",
}


RESULTS_DIR = "behavioural_analysis/results"

FONT_SETS = {
    "48pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_48",
    "72pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_72",
    "90pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_90",
    "108pt": "behavioural_analysis/visual_manipulations/font_size_manipulation/stroop_images_base_108",
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# MODEL LOADING
# ==========================
print(f"🔹 Loading CLIP model: {model_id}")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "brown", "pink", "gray", "black",
]

PROMPT_TEMPLATE = "The word is written in {} font."

# ==========================
# CORE EVALUATION FUNCTION
# ==========================
def evaluate_stroop_folder(stroop_dir, output_csv):
    """
    Evaluates all Stroop images in a given folder and saves results.
    """
    if not os.path.isdir(stroop_dir):
        print(f"⚠️ Skipping (missing folder): {stroop_dir}")
        return None

    results = []

    for word in tqdm(COLOR_WORDS, desc=f"Processing {stroop_dir}"):
        folder = os.path.join(stroop_dir, word)
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if not filename.endswith(".png"):
                continue

            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert("RGB")

            # Build prompts
            prompts = [PROMPT_TEMPLATE.format(color) for color in COLOR_WORDS]
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0]
                probs = logits.softmax(dim=0)
                top_idx = torch.argmax(probs).item()

            predicted_color = COLOR_WORDS[top_idx]
            font_color = filename.split("_as_")[1].replace(".png", "")

            results.append({
                "image": filename,
                "word_text": word,
                "font_color": font_color,
                "predicted_color": predicted_color,
                "is_word_prediction": predicted_color == word,
                "is_color_prediction": predicted_color == font_color,
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved results to {output_csv}")
    return df


# ==========================
# SUMMARY FUNCTION
# ==========================
def summarize_results(df, name):
    total = len(df)
    text_acc = (df["is_word_prediction"].sum() / total) * 100
    color_acc = (df["is_color_prediction"].sum() / total) * 100

    print(f"\n=== {name} ===")
    print(f"Total Images: {total}")
    print(f"Text Match Accuracy:  {text_acc:.2f}%")
    print(f"Color Match Accuracy: {color_acc:.2f}%")

    if text_acc > color_acc:
        conclusion = "🧠 Focus: text (Stroop effect)"
    elif color_acc > text_acc:
        conclusion = "🎨 Focus: color (visual bias)"
    else:
        conclusion = "⚖️ No clear bias"
    print(conclusion)

    return {
        "font_size": name,
        "n_images": total,
        "text_acc_%": round(text_acc, 2),
        "color_acc_%": round(color_acc, 2),
        "conclusion": conclusion,
    }


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    summaries = []

    for label, path in FONT_SETS.items():
        output_csv = os.path.join(RESULTS_DIR, f"clip_stroop_results_{label}.csv")
        df = evaluate_stroop_folder(path, output_csv)
        if df is not None and len(df):
            s = summarize_results(df, label)
            summaries.append(s)

    # Save summary table
    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(RESULTS_DIR, "clip_visual_fontsize_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\n📊 Summary saved to", summary_csv)
    print(summary_df)


if __name__ == "__main__":
    main()
