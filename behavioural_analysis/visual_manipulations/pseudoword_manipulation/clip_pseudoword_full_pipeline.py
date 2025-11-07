"""
CLIP Pseudoword Stroop — Full Pipeline
--------------------------------------
This unified script:
1. Generates pseudoword Stroop stimuli
2. Runs CLIP predictions on each color folder
3. Analyzes congruent/incongruent accuracy
4. Merges all summaries into one master CSV

Final Output:
    behavioural_analysis/pseudoword_analysis/results/
        ├── clip_predictions_<color>.csv
        ├── summarized_predictions/<color>_summary.csv
        └── merged_clip_prediction_summary.csv
"""

import os
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.join(os.getcwd(), "behavioural_analysis", "pseudoword_analysis")
STIMULI_DIR = os.path.join(BASE_DIR, "stroop_pseudowords_only")
SUMMARY_DIR = os.path.join(BASE_DIR, "summarized_predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(STIMULI_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = ["red","blue","green","yellow","orange","purple","brown","pink","gray","black"]

# ==========================
# STEP 1 — GENERATE PSEUDOWORDS
# ==========================
def generate_pseudoword_stimuli():
    print("\n🧩 Generating pseudoword Stroop stimuli...")
    color_map = {
        "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 128, 0),
        "yellow": (255, 255, 0), "orange": (255, 165, 0),
        "purple": (128, 0, 128), "brown": (139, 69, 19),
        "pink": (255, 192, 203), "gray": (128, 128, 128), "black": (0, 0, 0)
    }
    pseudowords_dict = {
        "red":["rebb","radz","rind","reld","ruxx"], "blue":["blum","blir","boop","blen","blak"],
        "green":["grel","grup","gron","gnek","griv"], "yellow":["yelm","yarp","yook","yuzz","yong"],
        "orange":["orfi","orak","orny","obex","orid"], "purple":["purx","purb","parv","puxo","pulv"],
        "brown":["brok","brun","braw","bork","bruz"], "pink":["ping","ponk","peet","pisk","pint"],
        "gray":["graw","gruz","grym","garn","greb"], "black":["blin","blan","blap","blor","blep"]
    }

    font_size, image_size, bg_color = 72, (384, 256), (255, 255, 255)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        print("⚠️ Using default font.")

    for word, color_rgb in color_map.items():
        word_dir = os.path.join(STIMULI_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        for pseudo in pseudowords_dict[word]:
            # congruent
            img = Image.new("RGB", image_size, bg_color)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), pseudo.upper(), font=font)
            pos = ((image_size[0]-bbox[2])//2, (image_size[1]-bbox[3])//2)
            draw.text(pos, pseudo.upper(), font=font, fill=color_rgb)
            img.save(os.path.join(word_dir, f"{word}_{pseudo}_congruent.png"))

            # incongruent
            for alt_color, alt_rgb in color_map.items():
                if alt_color == word: continue
                img = Image.new("RGB", image_size, bg_color)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), pseudo.upper(), font=font)
                pos = ((image_size[0]-bbox[2])//2, (image_size[1]-bbox[3])//2)
                draw.text(pos, pseudo.upper(), font=font, fill=alt_rgb)
                img.save(os.path.join(word_dir, f"{word}_{pseudo}_incongruent_{alt_color}.png"))
    print(f"✅ Stimuli saved to: {STIMULI_DIR}")


# ==========================
# STEP 2 — RUN CLIP PREDICTIONS
# ==========================
def run_clip_predictions():
    print("\n🤖 Running CLIP model on pseudoword stimuli...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    prompt_template = "The word is written in {} font."

    for color in tqdm(COLORS):
        folder = os.path.join(STIMULI_DIR, color)
        if not os.path.isdir(folder):
            continue

        results = []
        for filename in os.listdir(folder):
            if not filename.endswith(".png"):
                continue
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert("RGB")
            prompts = [prompt_template.format(c) for c in COLORS]
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits_per_image
                probs = logits.softmax(dim=1)[0]
                predicted_color = COLORS[probs.argmax().item()]

            results.append({"image": filename, "predicted_color": predicted_color})

        df = pd.DataFrame(results)
        output_path = os.path.join(RESULTS_DIR, f"clip_predictions_{color}.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved for {color}: {output_path}")


# ==========================
# STEP 3 — ANALYZE RESULTS
# ==========================
def analyze_results():
    print("\n📊 Analyzing CLIP pseudoword predictions...")
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    for file in os.listdir(RESULTS_DIR):
        if not file.startswith("clip_predictions_") or not file.endswith(".csv"):
            continue
        filepath = os.path.join(RESULTS_DIR, file)
        df = pd.read_csv(filepath)
        base_color = file.split("_")[2].replace(".csv", "").lower()

        summary = {"congruent_correct": 0, "color_correct": 0, "text_biased": 0, "off_target": 0, "total": 0}

        for _, row in df.iterrows():
            image = row["image"]
            pred = row["predicted_color"].strip().lower()

            if "incongruent" in image:
                font_color = image.split("_")[-1].replace(".png", "")
                if pred == font_color:
                    summary["color_correct"] += 1
                elif pred == base_color or pred in image:
                    summary["text_biased"] += 1
                else:
                    summary["off_target"] += 1
            elif "congruent" in image:
                if pred == base_color:
                    summary["congruent_correct"] += 1
            summary["total"] += 1

        summary_df = pd.DataFrame([summary])
        output_path = os.path.join(SUMMARY_DIR, file.replace(".csv", "_summary.csv"))
        summary_df.to_csv(output_path, index=False)
        print(f"✅ Summary saved: {output_path}")


# ==========================
# STEP 4 — MERGE SUMMARIES
# ==========================
def merge_summaries():
    print("\n🔗 Merging summaries...")
    summary_files = [f for f in os.listdir(SUMMARY_DIR) if f.endswith("_summary.csv")]
    all_summaries = []

    for file in summary_files:
        df = pd.read_csv(os.path.join(SUMMARY_DIR, file))
        color = file.split("_")[2]
        df.insert(0, "color", color)
        all_summaries.append(df)

    merged = pd.concat(all_summaries, ignore_index=True)
    merged_out = os.path.join(RESULTS_DIR, "merged_clip_prediction_summary.csv")
    merged.to_csv(merged_out, index=False)
    print(f"✅ Merged summary saved to: {merged_out}")
    print(merged)


# ==========================
# MAIN PIPELINE
# ==========================
if __name__ == "__main__":
    generate_pseudoword_stimuli()
    run_clip_predictions()
    analyze_results()
    merge_summaries()
    print("\n🎉 Full pseudoword Stroop pipeline completed successfully.")
