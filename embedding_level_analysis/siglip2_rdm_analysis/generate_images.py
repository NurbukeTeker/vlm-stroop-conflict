import os
from PIL import Image, ImageDraw, ImageFont

# === CONFIGURATION ===
base_dir = "stroop_images"
os.makedirs(base_dir, exist_ok=True)

# Paper-based RGB color codes
color_map = {
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

# Font settings
font_size = 90
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    font = ImageFont.load_default()
    print("⚠️ Arial font not found. Using default font.")

# Image size
image_size = (384, 256)

# === 1️⃣ Black text images ===
os.makedirs("images_black_text", exist_ok=True)
for word in color_map:
    img = Image.new("RGB", image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = word.upper()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    pos = ((image_size[0] - w)//2, (image_size[1] - h)//2)
    draw.text(pos, text, font=font, fill=(0, 0, 0))
    img.save(f"images_black_text/{word.upper()}_BLACK.png")

print("✅ Black text images saved → images_black_text/")

# === 2️⃣ Color-only patches ===
os.makedirs("images_color_only", exist_ok=True)
for name, rgb in color_map.items():
    img = Image.new("RGB", image_size, color=rgb)
    img.save(f"images_color_only/{name.upper()}_PATCH.png")
print("✅ Color-only patches saved → images_color_only/")

# === 3️⃣ Stroop images (congruent + incongruent) ===
for word in color_map:
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    text = word.upper()
    # Congruent
    img = Image.new("RGB", image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    pos = ((image_size[0]-w)//2, (image_size[1]-h)//2)
    draw.text(pos, text, font=font, fill=color_map[word])
    img.save(os.path.join(word_dir, f"{word}_as_{word}.png"))

    # Incongruent
    for ink, rgb in color_map.items():
        if ink == word:
            continue
        img = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, font=font, fill=rgb)
        img.save(os.path.join(word_dir, f"{word}_as_{ink}.png"))

print(f"✅ Stroop congruent/incongruent images saved → {base_dir}/")
print("🎉 All 3 sets (black_text, color_only, stroop) generated successfully!")
