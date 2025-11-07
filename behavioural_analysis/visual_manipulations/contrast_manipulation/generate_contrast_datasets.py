"""
Generate Stroop Contrast Manipulation Datasets (Full Version)
-------------------------------------------------------------
Creates both congruent and incongruent Stroop-style datasets
with multiple background contrast levels (high, medium, low, same).

Output structure:
    behavioural_analysis/
        visual_manipulations/
            contrast_manipulation/
                stroop_contrast_variants/
                stroop_incongruent_contrast_variants/
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.join(os.getcwd(), "behavioural_analysis", "visual_manipulations", "contrast_manipulation")

CONGRUENT_DIR = os.path.join(BASE_DIR, "stroop_contrast_variants")
INCONGRUENT_DIR = os.path.join(BASE_DIR, "stroop_incongruent_contrast_variants")

os.makedirs(CONGRUENT_DIR, exist_ok=True)
os.makedirs(INCONGRUENT_DIR, exist_ok=True)

# Color map for text colors (RGB)
COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (139, 69, 19),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "black": (0, 0, 0)
}

# ==========================
# CONTRAST LEVEL FUNCTIONS
# ==========================
def high_contrast(_):
    return (255, 255, 255)  # White background

def medium_contrast(color):
    return tuple(min(255, int(c + (255 - c) * 0.5)) for c in color)

def low_contrast(color):
    return tuple(int(c * 0.8) for c in color)

def same_color(color):
    return color

CONTRAST_LEVELS = [
    ("high", high_contrast),
    ("medium", medium_contrast),
    ("low", low_contrast),
    ("same", same_color)
]

# ==========================
# FONT SETTINGS
# ==========================
FONT_SIZE = 90
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()
    print("⚠️ Arial font not found. Using default PIL font instead.")

IMAGE_SIZE = (384, 256)

# ==========================
# GENERATE CONGRUENT IMAGES
# ==========================
def generate_congruent():
    print("\n🧩 Generating CONGRUENT contrast variants...")
    for word, font_color in COLOR_MAP.items():
        word_dir = os.path.join(CONGRUENT_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        for level_name, bg_func in CONTRAST_LEVELS:
            bg_color = bg_func(font_color)

            img = Image.new("RGB", IMAGE_SIZE, color=bg_color)
            draw = ImageDraw.Draw(img)

            text = word.upper()
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            position = ((IMAGE_SIZE[0] - text_w) // 2, (IMAGE_SIZE[1] - text_h) // 2)

            draw.text(position, text, font=font, fill=font_color)
            filename = f"{word}_bg_{level_name}.png"
            img.save(os.path.join(word_dir, filename))

    print(f"✅ Congruent images saved to: {CONGRUENT_DIR}")


# ==========================
# GENERATE INCONGRUENT IMAGES
# ==========================
def generate_incongruent():
    print("\n🧩 Generating INCONGRUENT contrast variants...")
    for word, _ in COLOR_MAP.items():
        word_dir = os.path.join(INCONGRUENT_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        for font_color_name, font_rgb in COLOR_MAP.items():
            if font_color_name == word:
                continue  # skip congruent cases

            for level_name, bg_func in CONTRAST_LEVELS:
                bg_color = bg_func(font_rgb)

                img = Image.new("RGB", IMAGE_SIZE, color=bg_color)
                draw = ImageDraw.Draw(img)

                text = word.upper()
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                position = ((IMAGE_SIZE[0] - text_w) // 2, (IMAGE_SIZE[1] - text_h) // 2)

                draw.text(position, text, font=font, fill=font_rgb)
                filename = f"{word}_as_{font_color_name}_bg_{level_name}.png"
                img.save(os.path.join(word_dir, filename))

    print(f"✅ Incongruent images saved to: {INCONGRUENT_DIR}")


# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    generate_congruent()
    generate_incongruent()
    print("\n🎉 All Stroop contrast datasets generated successfully.")
