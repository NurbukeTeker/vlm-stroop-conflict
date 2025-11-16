import os
import json
import torch
from torch.nn.functional import cosine_similarity

# ============================================================
# PATHS
# ============================================================
EMBED_ROOT = "saved/embeddings/congruent"
CHUNK_COLOR_ROOT = "saved/chunks/color"
CHUNK_TEXT_ROOT = "saved/chunks/text"

SAVE_PATH = "saved/results/steering_cosine.json"

TARGET_COLORS = ["red","blue","green","yellow","orange","purple","pink","brown","gray","black"]


# ============================================================
# LOAD TEXT PROMPT EMBEDDINGS
# (Computed once using Qwen model or provided externally)
# But here: we recompute from chunks dictionary (no Qwen needed)
# ============================================================
def build_dummy_text_embeddings():
    """
    We need target embeddings to compute cosine similarities.
    Here we create a distinct embedding per color/word to have consistency.
    If you want actual Qwen text embeddings, you can plug them in.
    """
    emb_dict = {}
    torch.manual_seed(42)
    for c in TARGET_COLORS:
        vec = torch.randn(4096)  # Qwen2-VL hidden size
        vec = vec / vec.norm()
        emb_dict[c] = vec
    return emb_dict


text_emb_color = build_dummy_text_embeddings()
text_emb_word  = build_dummy_text_embeddings()


# ============================================================
# LOAD LAYERWISE EMBEDDINGS
# ============================================================
def load_congruent_embeddings():
    """
    Returns structure:
    {
      'red_as_red.png': { layer4: tensor, layer16: tensor, ... }
    }
    """
    print("[INFO] Loading congruent embeddings...")
    data = {}

    for lbl in sorted(os.listdir(EMBED_ROOT)):
        lbl_dir = os.path.join(EMBED_ROOT, lbl)
        if not os.path.isdir(lbl_dir):
            continue

        for layer_folder in os.listdir(lbl_dir):
            if not layer_folder.startswith("layer"):
                continue

            layer_idx = int(layer_folder.replace("layer", ""))
            layer_dir = os.path.join(lbl_dir, layer_folder)

            for f in os.listdir(layer_dir):
                if not f.endswith(".pt"):
                    continue

                full_path = os.path.join(layer_dir, f)
                emb = torch.load(full_path, map_location="cpu").squeeze(0)

                data.setdefault(f, {})[layer_idx] = emb

    print("[INFO] Loaded", len(data), "congruent images.")
    return data


# ============================================================
# LOAD CHUNKS (COLOR AND TEXT)
# ============================================================
def load_chunks(root_dir):
    """
    Return structure:
    {
      layer_idx : { 'red': tensor, 'blue': tensor, ... }
    }
    """
    layers = {}

    for f in os.listdir(root_dir):
        if f.endswith("_all.pt"):
            layer_idx = int(f.replace("layer", "").replace("_all.pt", ""))
            layers[layer_idx] = torch.load(
                os.path.join(root_dir, f), map_location="cpu"
            )

    return layers


# ============================================================
# APPLY STEERING
# ============================================================
def apply_chunk(original_vec, chunk_vec):
    new_vec = original_vec + chunk_vec
    return new_vec / new_vec.norm()


# ============================================================
# MAIN STEERING EVALUATION
# ============================================================
def steering_evaluation():
    print("[INFO] Starting steering evaluation...")

    congruent = load_congruent_embeddings()
    color_chunks = load_chunks(CHUNK_COLOR_ROOT)
    text_chunks  = load_chunks(CHUNK_TEXT_ROOT)

    results = []

    for filename, layer_dict in congruent.items():
        print(f"[INFO] Processing {filename}")

        # Parse original label → "red_as_red.png"
        name = filename.replace(".pt", "")
        word, _, color = name.partition("_as_")

        for layer_idx, emb in layer_dict.items():

            if layer_idx not in color_chunks or layer_idx not in text_chunks:
                continue

            color_chunk_map = color_chunks[layer_idx]
            text_chunk_map = text_chunks[layer_idx]

            for target in TARGET_COLORS:
                if target == color:
                    continue  # steering only when target != original

                # =====================================================
                # COLOR STEERING
                # =====================================================
                if color in color_chunk_map and target in color_chunk_map:
                    chunk_color = color_chunk_map[target] - color_chunk_map[color]
                    chunk_color = chunk_color / chunk_color.norm()

                    steered_color = apply_chunk(emb, chunk_color)

                    before_c = cosine_similarity(
                        emb.unsqueeze(0), text_emb_color[color].unsqueeze(0)
                    ).item()
                    after_c = cosine_similarity(
                        steered_color.unsqueeze(0), text_emb_color[target].unsqueeze(0)
                    ).item()
                else:
                    before_c = None
                    after_c = None

                # =====================================================
                # TEXT STEERING
                # =====================================================
                if word in text_chunk_map and target in text_chunk_map:
                    chunk_text = text_chunk_map[target] - text_chunk_map[word]
                    chunk_text = chunk_text / chunk_text.norm()

                    steered_text = apply_chunk(emb, chunk_text)

                    before_w = cosine_similarity(
                        emb.unsqueeze(0), text_emb_word[word].unsqueeze(0)
                    ).item()
                    after_w = cosine_similarity(
                        steered_text.unsqueeze(0), text_emb_word[target].unsqueeze(0)
                    ).item()
                else:
                    before_w = None
                    after_w = None

                results.append({
                    "filename": filename,
                    "layer": layer_idx,
                    "source_word": word,
                    "source_color": color,
                    "target": target,
                    "delta_color": None if before_c is None else (after_c - before_c),
                    "delta_word": None  if before_w is None else (after_w - before_w),
                })

    print("[INFO] Saving results...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("[INFO] Steering evaluation completed.")
    print("[INFO] Results saved →", SAVE_PATH)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    steering_evaluation()
