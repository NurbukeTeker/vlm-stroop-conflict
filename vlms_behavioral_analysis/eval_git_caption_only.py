import os
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from vlm_shared import load_stroop_images, parse_filename, save_results_csv

# =======================================================
# MODEL LOAD
# =======================================================

MODEL_ID = "microsoft/git-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()

COLORS = ["red","blue","green","yellow","orange","purple","brown","pink","gray","black"]


# =======================================================
# CAPTION GENERATION
# =======================================================

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    caption = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return caption.strip().lower()


# =======================================================
# CAPTION COLOR EXTRACTION
# =======================================================

def detect_colors_in_caption(caption):
    found = []
    for c in COLORS:
        if re.search(rf"\b{c}\b", caption):
            found.append(c)
    return found  # could be multiple


# =======================================================
# CLASSIFY OUTCOME: Ink / Word / Both / Neither
# =======================================================

def classify(word, ink, found_colors):
    has_word = (word in found_colors)
    has_ink = (ink in found_colors)

    if has_word and has_ink:
        return "both"
    elif has_ink:
        return "ink"
    elif has_word:
        return "word"
    else:
        return "neither"


# =======================================================
# MAIN LOOP
# =======================================================

def main():
    image_paths = load_stroop_images()
    rows = []

    for path in tqdm(image_paths, desc="GIT caption analysis"):
        
        norm_path = os.path.normpath(path)
        fname = os.path.basename(norm_path)

        # parse from filename
        word, ink, condition = parse_filename(fname)
        if word is None:
            continue

        # generate caption
        caption = generate_caption(norm_path)

        # extract colors
        found_colors = detect_colors_in_caption(caption)

        # classify type
        outcome = classify(word, ink, found_colors)

        rows.append([
            fname, word, ink, condition,
            caption, ",".join(found_colors), outcome
        ])

    save_results_csv(rows, "results_git_caption_analysis.csv")
    print("\nSaved → results_git_caption_analysis.csv")

if __name__ == "__main__":
    main()
