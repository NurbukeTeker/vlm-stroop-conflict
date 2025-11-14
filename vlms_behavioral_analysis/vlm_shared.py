import os
import re
import csv
from PIL import Image

# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_DIR = os.path.join("..", "embedding_level_analysis", "stroop_images")

# ============================================================
# 2. PROMPTS FOR GENERATIVE VLMS
# ============================================================

PROMPTS = {
    "short": "What color is the word in this image?",

    "strong": (
        "You will see an English word written in a colored ink. "
        "Ignore the written word and answer ONLY the ink color. "
        "Respond with one lowercase color name: red, blue, green, "
        "yellow, orange, purple, pink, brown, gray, or black."
    ),

    "mc": (
        "What is the ink color of the text in this image? "
        "Choose one of the following: red, blue, green, yellow, orange, "
        "purple, pink, brown, gray, black. Answer using only one of these color words."
    ),
}

# ============================================================
# 3. COLOR VOCABULARY & SYNONYMS
# ============================================================

CANONICAL_COLORS = [
    "black","blue","brown","gray","green",
    "orange","pink","purple","red","yellow"
]

COLOR_SYNONYMS = {
    "scarlet": "red",
    "crimson": "red",
    "maroon": "red",
    "azure": "blue",
    "navy": "blue",
    "skyblue": "blue",
    "lime": "green",
    "olive": "green",
    "gold": "yellow",
    "violet": "purple",
    "magenta": "pink",
    "rose": "pink",
    "beige": "brown",
    "tan": "brown",
    "charcoal": "gray",
    "silver": "gray",
}

def normalize_text(s):
    """Lowercase, remove punctuation, map synonyms."""
    if s is None:
        return ""

    s = s.lower().strip()
    s = re.sub(r"[^a-z ]+", "", s)  # remove punctuation
    s = s.replace("colour", "color")

    # Map synonyms
    for syn, canon in COLOR_SYNONYMS.items():
        if syn in s:
            return canon

    # direct match
    for c in CANONICAL_COLORS:
        if c in s:
            return c

    return s  # fallback → may be a weird unmatched word


# ============================================================
# 4. LABEL MAPPING
# ============================================================

def classify_output(output, word, ink):
    """
    Map model output to one of:
    - Ink Match
    - Word Match
    - Both
    - Neither
    """

    norm = normalize_text(output)

    ink_match = (ink in norm)
    word_match = (word in norm)

    if ink_match and word_match:
        return "both"
    if ink_match:
        return "ink"
    if word_match:
        return "word"
    return "neither"


# ============================================================
# 5. FILENAME PARSER
# ============================================================

def parse_filename(fname):
    """
    Example filename:
        RED_as_blue.png
    or from augmented versions:
        RED_as_blue_bg-softpink_...
    """
    fname = fname.lower()

    if "_as_" not in fname:
        return None, None, None

    word = fname.split("_as_")[0]
    ink  = fname.split("_as_")[1].split("_")[0]

    condition = "congruent" if word == ink else "incongruent"

    return word, ink, condition


# ============================================================
# 6. LOAD STROOP SMALL DATASET (100 IMAGES)
# ============================================================

def load_stroop_images():
    paths = []

    for color in os.listdir(DATASET_DIR):
        color_dir = os.path.join(DATASET_DIR, color)
        if not os.path.isdir(color_dir):
            continue

        for f in os.listdir(color_dir):
            if f.endswith(".png"):
                paths.append(os.path.join(color_dir, f))

    return paths


# ============================================================
# 7. SAVE CSV
# ============================================================

def save_results_csv(rows, outname):
    with open(outname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "word", "ink", "condition",
            "output", "label"
        ])
        writer.writerows(rows)

    print(f"Saved → {outname}")
