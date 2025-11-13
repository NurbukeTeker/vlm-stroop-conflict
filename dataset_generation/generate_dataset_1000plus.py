import os
import csv
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# ============================================================
# OUTPUT STRUCTURE
# ============================================================

BASE_DIR = "stroop_dataset_v2"
CONG_DIR = os.path.join(BASE_DIR, "congruent")
INCONG_DIR = os.path.join(BASE_DIR, "incongruent")

os.makedirs(CONG_DIR, exist_ok=True)
os.makedirs(INCONG_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMG_W, IMG_H = 384, 256
FONT_SIZE = 90

try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except:
    FONT = ImageFont.load_default()
    print("⚠️ Arial font not found. Using default font.")

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

BACKGROUND_COLORS = {
    (255, 255, 255): "white",
    (235, 235, 235): "softgray",
    (245, 240, 225): "beige",
    (250, 230, 235): "softpink",
    (230, 245, 235): "mint",
    (250, 250, 225): "softyellow",
}

TONE_VARIANTS = [0.6, 0.8, 1.0, 1.2, 1.4]
SAT_VARIANTS  = [0.8, 1.0, 1.2]
BRI_VARIANTS  = [0.8, 1.0, 1.2]

MIN_CONTRAST = 40

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_tone(rgb, factor):
    r, g, b = rgb
    return (
        min(int(r * factor), 255),
        min(int(g * factor), 255),
        min(int(b * factor), 255),
    )

def rgb_distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5

def enc(x):
    """Encode 0.8 → 08, 1.0 → 10, etc."""
    return f"{int(x * 10):02d}"

def draw_centered_text(img, text, ink_rgb):
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=FONT)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    pos = ((IMG_W - w)//2, (IMG_H - h)//2)
    draw.text(pos, text, fill=ink_rgb, font=FONT)

# ============================================================
# METADATA
# ============================================================

metadata = []

# ============================================================
# 1) CONGRUENT DATA (minimal only 10 images)
# ============================================================

print("Generating CONGRUENT samples...")

for word, rgb in BASE_COLORS.items():

    text = word.upper()

    img = Image.new("RGB", (IMG_W, IMG_H), (255,255,255))  # white background
    draw_centered_text(img, text, rgb)

    fname = f"{text}_as_{word}_bg-white.png"
    img.save(os.path.join(CONG_DIR, fname))

    metadata.append([fname, word, word, "congruent"])

print("✔ Done: 10 congruent samples.")


# ============================================================
# 2) INCONGRUENT DATA (fully augmented)
# ============================================================

print("Generating INCONGRUENT samples...")

for word, base_rgb in BASE_COLORS.items():

    for ink_color, ink_rgb_base in BASE_COLORS.items():
        if ink_color == word:
            continue  # skip congruent

        for tone in TONE_VARIANTS:
            ink_rgb = apply_tone(ink_rgb_base, tone)
            tone_code = enc(tone)

            for bg_rgb, bg_name in BACKGROUND_COLORS.items():

                if rgb_distance(ink_rgb, bg_rgb) < MIN_CONTRAST:
                    continue

                for sat in SAT_VARIANTS:
                    sat_code = enc(sat)

                    for bri in BRI_VARIANTS:
                        bri_code = enc(bri)

                        img = Image.new("RGB", (IMG_W, IMG_H), bg_rgb)
                        img = ImageEnhance.Color(img).enhance(sat)
                        img = ImageEnhance.Brightness(img).enhance(bri)

                        text = word.upper()
                        draw_centered_text(img, text, ink_rgb)

                        fname = (
                            f"{text}_as_{ink_color}_bg-{bg_name}"
                            f"_T{tone_code}_S{sat_code}_B{bri_code}.png"
                        )

                        img.save(os.path.join(INCONG_DIR, fname))

                        metadata.append([
                            fname, word, ink_color, "incongruent"
                        ])

print("✔ Finished INCONGRUENT dataset.")


# ============================================================
# SAVE METADATA
# ============================================================

with open(os.path.join(BASE_DIR, "metadata.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "word", "ink", "condition"])
    w.writerows(metadata)

print("===================================================")
print(" ALL DONE!")
print(" Total images:", len(metadata))
print("===================================================")
