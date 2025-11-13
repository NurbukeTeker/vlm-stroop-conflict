import os
import csv
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "stroop_dataset_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classic Stroop image size + font
IMG_W, IMG_H = 384, 256
FONT_SIZE = 90

try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except:
    FONT = ImageFont.load_default()
    print("⚠️ Arial font not found, using default.")

# Paper-based RGB colors
BASE_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (139, 69, 19),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
}

# 6 pastel backgrounds
BACKGROUND_COLORS = {
    (255, 255, 255): "white",
    (235, 235, 235): "softgray",
    (245, 240, 225): "beige",
    (250, 230, 235): "softpink",
    (230, 245, 235): "mint",
    (250, 250, 225): "softyellow",
}

# Tone multipliers
TONE_VARIANTS = [0.6, 0.8, 1.0, 1.2, 1.4]

# Perceptual augmentations
SAT_VARIANTS = [0.8, 1.0, 1.2]
BRI_VARIANTS = [0.8, 1.0, 1.2]

# Minimum contrast threshold
MIN_CONTRAST = 40


# ============================================================
# HELPERS
# ============================================================

def apply_tone(rgb, factor):
    r, g, b = rgb
    return (min(int(r * factor), 255),
            min(int(g * factor), 255),
            min(int(b * factor), 255))

def rgb_distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5

def encode_factor(value):
    """Convert 0.6 → 06, 1.0 → 10, etc."""
    return f"{int(value * 10):02d}"


# ============================================================
# GENERATION LOOP
# ============================================================

metadata = []

for word, base_rgb in BASE_COLORS.items():

    for ink_tone in TONE_VARIANTS:
        ink_rgb = apply_tone(base_rgb, ink_tone)
        ink_tone_code = encode_factor(ink_tone)

        for bg_rgb, bg_name in BACKGROUND_COLORS.items():

            # Ensure contrast
            if rgb_distance(ink_rgb, bg_rgb) < MIN_CONTRAST:
                continue

            for sat in SAT_VARIANTS:
                sat_code = encode_factor(sat)

                for bri in BRI_VARIANTS:
                    bri_code = encode_factor(bri)

                    # Create fresh image background
                    img = Image.new("RGB", (IMG_W, IMG_H), bg_rgb)

                    # Apply saturation + brightness
                    img = ImageEnhance.Color(img).enhance(sat)
                    img = ImageEnhance.Brightness(img).enhance(bri)

                    # Draw text
                    draw = ImageDraw.Draw(img)
                    text = word.upper()

                    # Pillow 10+ safe bbox
                    bbox = draw.textbbox((0, 0), text, font=FONT)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    pos = ((IMG_W - text_w) // 2, (IMG_H - text_h) // 2)

                    draw.text(pos, text, fill=ink_rgb, font=FONT)

                    # Filename
                    filename = (
                        f"{text}_as_{word}_bg-{bg_name}"
                        f"_T{ink_tone_code}_S{sat_code}_B{bri_code}.png"
                    )

                    filepath = os.path.join(OUTPUT_DIR, filename)
                    img.save(filepath)

                    # Save metadata
                    metadata.append([
                        filename,
                        word,
                        ink_rgb,
                        bg_rgb,
                        ink_tone,
                        sat,
                        bri
                    ])


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
with open(metadata_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "word",
        "ink_rgb",
        "background_rgb",
        "tone_factor",
        "saturation",
        "brightness"
    ])
    writer.writerows(metadata)

print("DONE!")
print("Total images:", len(metadata))
