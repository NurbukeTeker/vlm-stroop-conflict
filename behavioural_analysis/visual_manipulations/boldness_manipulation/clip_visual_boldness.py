"""
CLIP Visual Manipulation — Boldness Analysis
--------------------------------------------
Evaluates CLIP model robustness across text **boldness variations**
(light / normal / bold / narrow) using Stroop-style color–word conflict images
(all rendered at 90 pt font size).

Folder structure expected:
    behavioural_analysis/
        visual_manipulations/
            boldness_manipulation/
                stroop_images_bold_light/
                stroop_images_bold_normal/
                stroop_images_bold_bold/
                stroop_images_bold_narrow/
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
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
prompt_template = "The word is written in {} font."

BASE_DIR = "behavioural_analysis/visual_manipulations/boldness_manipulation"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BOLDNESS_DIRS = {
    "light": os.path.join(BASE_DIR, "stroop_images_bold_light"),
    "normal": os.path.join(BASE_DIR, "stroop_images_bold_normal"),
    "bold": os.path.join(BASE_DIR, "stroop_images_bold_bold"),
    "narrow": os.path.join(BASE_DIR, "stroop_images_bold_narrow"),
}

COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "brown", "pink", "gray", "black"
]

# ==========================
# MODEL LOADING
# ==========================
print(f"🔹 Loading CLIP model: {model_id}")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

# ==========================
# CORE FUNCTION
# ==========================
def evaluate_boldness(stroop_dir, out_csv):
    """
    Evaluate one boldness folder and save per-image results.
    """
    if not os.path.isdir(stroop_dir):
        print(f"⚠️ Skipping missing folder: {stroop_dir}")
        return None

    results = []

    for word in tqdm(COLOR_WORDS, desc=f"Scanning {os.path.basename(stroop_dir)}"):
        folder = os.path.join(stroop_dir, word)
        if not os.path.isdir(folder):
            continue

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".png"):
                continue

            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert("RGB")

            prompts = [prompt_template.format(c) for c in COLOR_WORDS]
            inputs = processor(text=prompts, images=image,
                               return_tensors="pt", padding=True).to(device)

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
                "is_color_prediction": predicted_color == font_color
            })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved results to {out_csv}")
    return df


# ==========================
# SUMMARY
# ==========================
def summarize_results(label, df):
    total = len(df)
    text_acc = (df["is_word_prediction"].sum() / total) * 100
    color_acc = (df["is_color_prediction"].sum() / total) * 100

    if text_acc > color_acc:
        conclusion = "🧠 Focus: text (Stroop effect)"
    elif color_acc > text_acc:
        conclusion = "🎨 Focus: color (visual bias)"
    else:
        conclusion = "⚖️ No clear bias"

    print(f"\n=== {label.upper()} ===")
    print(f"Total: {total}")
    print(f"Text Accuracy:  {text_acc:.2f}%")
    print(f"Color Accuracy: {color_acc:.2f}%")
    print(conclusion)

    return {
        "Style": label,
        "n_images": total,
        "text_acc_%": round(text_acc, 2),
        "color_acc_%": round(color_acc, 2),
        "conclusion": conclusion,
    }


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    all_summaries = []

    for label, path in BOLDNESS_DIRS.items():
        out_csv = os.path.join(RESULTS_DIR, f"clip_stroop_results_bold_{label}.csv")
        df = evaluate_boldness(path, out_csv)
        if df is not None and len(df):
            summary = summarize_results(label, df)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(RESULTS_DIR, "clip_visual_boldness_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\n📊 Summary saved to", summary_csv)
    print(summary_df)


if __name__ == "__main__":
    main()
