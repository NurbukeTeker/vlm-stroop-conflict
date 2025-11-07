# Install necessary packages if not already installed
# !pip install torch torchvision torchaudio
# !pip install openai-clip
# !pip install umap-learn
# !pip install matplotlib
# !pip install scikit-learn

import torch
import clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import os
import json
from scipy.spatial import ConvexHull

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define storage paths
base_dir = "embedding_analysis"
os.makedirs(base_dir, exist_ok=True)

# Define colors for Stroop test images
colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gray", "black"]
color_map = {"red": "#FF0000", "blue": "#0000FF", "green": "#008000", "yellow": "#FFFF00", "orange": "#FFA500", "purple": "#800080", "brown": "#964B00", "pink": "#FFC0CB", "gray": "#808080", "black": "#000000"}

# Generate and store images
def create_stroop_images(word):
    word_dir = os.path.join(base_dir, word.upper())
    image_dir = os.path.join(word_dir, "stroop_images")
    os.makedirs(image_dir, exist_ok=True)
    image_text_pairs = []
    for color in colors:
        filename = os.path.join(image_dir, f"{word.lower()}_word_{color}.jpg")
        if not os.path.exists(filename):
            img = Image.new("RGB", (400, 200), "white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except:
                font = ImageFont.load_default()
            text_size = draw.textbbox((0, 0), word, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            draw.text(((img.width - text_width) // 2, (img.height - text_height) // 2), word, fill=color, font=font)
            img_gray = img.convert("L")
            bbox = img_gray.getbbox()
            if bbox:
                img = img.crop(bbox)
            img.save(filename)
        image_text_pairs.append((filename, f"The word '{word.upper()}' written in {color} color."))
    return word_dir, image_text_pairs

# Extract embeddings
def extract_embeddings(image_text_pairs):
    image_embeddings, text_embeddings, labels = [], [], []
    for img_path, text in image_text_pairs:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text_tokenized = clip.tokenize([text]).to(device)
        with torch.no_grad():
            image_embed = model.encode_image(image).cpu().numpy()
            text_embed = model.encode_text(text_tokenized).cpu().numpy()
        image_embeddings.append(image_embed)
        text_embeddings.append(text_embed)
        labels.append(text)
    return np.vstack(image_embeddings), np.vstack(text_embeddings), labels

# Process all words for congruent-incongruent analysis
all_embeddings = []
all_labels = []
all_congruent_indices = []
all_words = []
for word in colors:
    word_dir, image_text_pairs = create_stroop_images(word)
    image_embeddings, text_embeddings, labels = extract_embeddings(image_text_pairs)
    all_embeddings.append(np.vstack([image_embeddings, text_embeddings]))
    all_labels.append(labels)
    all_congruent_indices.append(colors.index(word))
    all_words.append(word)

# Combine all embeddings and apply a single UMAP transformation
all_embeddings = np.vstack(all_embeddings)
reducer = umap.UMAP(n_components=2, random_state=42)
low_dim_embeddings = reducer.fit_transform(all_embeddings)

# Plot all embeddings in one figure
plt.figure(figsize=(12, 8))
for i, word in enumerate(all_words):
    color = color_map[word]
    marker_v = "o"  # Marker for visual embeddings
    marker_t = "s"  # Marker for text embeddings

    # Get indices for congruent case and its related embeddings
    start_idx = i * len(colors) * 2  # Each word has both visual and text embeddings
    visual_embeddings = low_dim_embeddings[start_idx:start_idx + len(colors)]
    text_embeddings = low_dim_embeddings[start_idx + len(colors):start_idx + 2 * len(colors)]

    # Scatter points
    plt.scatter(visual_embeddings[:, 0], visual_embeddings[:, 1], color=color, marker=marker_v, label=f"{word} (Visual)", alpha=0.7)
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], color=color, marker=marker_t, label=f"{word} (Text)", alpha=0.7)

    # Connect visual-text embeddings
    for j in range(len(colors)):
        plt.plot([visual_embeddings[j, 0], text_embeddings[j, 0]],
                 [visual_embeddings[j, 1], text_embeddings[j, 1]],
                 color=color, alpha=0.5)

plt.title("CLIP Embedding Space Visualization for All Congruent and Incongruent Cases")
plt.legend()
plt.savefig(os.path.join(base_dir, "all_embeddings_combined_umap.png"))
plt.show()