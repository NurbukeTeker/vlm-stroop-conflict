import os
import json
import torch
from torch.nn.functional import cosine_similarity

# ============================================================
# PATHS (absolute)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

EMBED_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "saved", "embeddings", "congruent"))
CHUNK_COLOR_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "saved", "chunks_subpop", "color"))
CHUNK_TEXT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, "..", "saved", "chunks_subpop", "text"))
SAVE_PATH        = os.path.abspath(os.path.join(THIS_DIR, "..", "saved", "results", "steering_subpop.json"))

print(f"[INFO] EMBED_ROOT = {EMBED_ROOT}")
print(f"[INFO] CHUNK_COLOR_ROOT = {CHUNK_COLOR_ROOT}")
print(f"[INFO] CHUNK_TEXT_ROOT = {CHUNK_TEXT_ROOT}")
print(f"[INFO] SAVE_PATH = {SAVE_PATH}")

TARGET_COLORS = [
    "red","blue","green","yellow","orange",
    "purple","pink","brown","gray","black"
]

# ============================================================
# DUMMY TEXT EMBEDDINGS (use 1280 dims)
# ============================================================
def build_text_embeddings(dim=1280):
    emb = {}
    torch.manual_seed(42)
    for c in TARGET_COLORS:
        v = torch.randn(dim)
        emb[c] = v / v.norm()
    return emb

text_emb_color = build_text_embeddings(1280)
text_emb_word  = build_text_embeddings(1280)


# ============================================================
# LOAD CONGRUENT EMBEDDINGS
# ============================================================
def load_congruent_embeddings():

    print("[INFO] Loading congruent embeddings...")

    if not os.path.isdir(EMBED_ROOT):
        print(f"[ERROR] Congruent directory missing: {EMBED_ROOT}")
        return {}

    data = {}

    for layer_folder in sorted(os.listdir(EMBED_ROOT)):
        if not layer_folder.startswith("layer"):
            continue

        L = int(layer_folder.replace("layer", ""))
        layer_dir = os.path.join(EMBED_ROOT, layer_folder)

        for f in sorted(os.listdir(layer_dir)):
            if not f.endswith(".pt"):
                continue

            path = os.path.join(layer_dir, f)
            emb = torch.load(path, map_location="cpu")

            # Extract CLS token
            if emb.dim() == 2:
                emb = emb[0]
            elif emb.dim() != 1:
                raise ValueError(f"Unexpected emb shape {emb.shape} in {path}")

            emb = emb / (emb.norm() + 1e-12)
            data.setdefault(f, {})[L] = emb

    print(f"[INFO] Loaded {len(data)} congruent images.")
    return data


# ============================================================
# LOAD SUBPOP CHUNKS
# ============================================================
def load_chunks(root_dir):

    layers = {}

    if not os.path.isdir(root_dir):
        print(f"[WARNING] Missing chunk dir: {root_dir}")
        return layers

    for f in os.listdir(root_dir):
        if f.endswith("_all.pt"):
            L = int(f.replace("layer", "").replace("_all.pt", ""))
            layers[L] = torch.load(os.path.join(root_dir, f), map_location="cpu")

    return layers


# ============================================================
# APPLY STEERING (vector addition + renorm)
# ============================================================
def apply_chunk(original_vec, chunk_vec):
    new_vec = original_vec + chunk_vec
    return new_vec / (new_vec.norm() + 1e-12)


# ============================================================
# MAIN STEERING EVALUATION (NOW WITH COMBINED)
# ============================================================
def steering_evaluation():

    print("[INFO] Starting SUBPOPULATION steering evaluation...")

    congruent     = load_congruent_embeddings()
    color_chunks  = load_chunks(CHUNK_COLOR_ROOT)
    text_chunks   = load_chunks(CHUNK_TEXT_ROOT)

    results = []

    for filename, layer_dict in congruent.items():

        name = filename.replace(".pt", "")
        word, _, color = name.partition("_as_")

        for L, emb in layer_dict.items():

            if L not in color_chunks or L not in text_chunks:
                continue

            color_map = color_chunks[L]
            text_map  = text_chunks[L]

            for target in TARGET_COLORS:
                if target == color:
                    continue

                # -------------------------------------
                # 1) COLOR STEERING
                # -------------------------------------
                if color in color_map and target in color_map:

                    diff_color = color_map[target] - color_map[color]
                    diff_color = diff_color / (diff_color.norm() + 1e-12)

                    steered_color = apply_chunk(emb, diff_color)

                    before_c = cosine_similarity(
                        emb.unsqueeze(0),
                        text_emb_color[color].unsqueeze(0)
                    ).item()

                    after_c = cosine_similarity(
                        steered_color.unsqueeze(0),
                        text_emb_color[target].unsqueeze(0)
                    ).item()

                else:
                    before_c = None
                    after_c  = None

                # -------------------------------------
                # 2) TEXT STEERING
                # -------------------------------------
                if word in text_map and target in text_map:

                    diff_text = text_map[target] - text_map[word]
                    diff_text = diff_text / (diff_text.norm() + 1e-12)

                    steered_text = apply_chunk(emb, diff_text)

                    before_w = cosine_similarity(
                        emb.unsqueeze(0),
                        text_emb_word[word].unsqueeze(0)
                    ).item()

                    after_w = cosine_similarity(
                        steered_text.unsqueeze(0),
                        text_emb_word[target].unsqueeze(0)
                    ).item()

                else:
                    before_w = None
                    after_w  = None

                # -------------------------------------
                # 3) COMBINED STEERING
                # -------------------------------------
                if (color in color_map) and (word in text_map) and (target in color_map) and (target in text_map):

                    combined = (
                        - color_map[color]
                        - text_map[word]
                        + color_map[target]
                        + text_map[target]
                    )
                    combined = combined / (combined.norm() + 1e-12)

                    steered_combined = apply_chunk(emb, combined)

                    before_comb = cosine_similarity(
                        emb.unsqueeze(0),
                        text_emb_color[color].unsqueeze(0)
                    ).item()

                    after_comb = cosine_similarity(
                        steered_combined.unsqueeze(0),
                        text_emb_color[target].unsqueeze(0)
                    ).item()

                    delta_combined = after_comb - before_comb

                else:
                    delta_combined = None

                # -------------------------------------
                # SAVE
                # -------------------------------------
                results.append({
                    "filename": filename,
                    "layer": L,
                    "source_word": word,
                    "source_color": color,
                    "target": target,
                    "delta_color": None if before_c is None else (after_c - before_c),
                    "delta_word":  None if before_w is None else (after_w - before_w),
                    "delta_combined": delta_combined
                })

    # =======================================================
    # SAVE RESULTS
    # =======================================================
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("[INFO] SUBPOP + COMBINED steering evaluation DONE.")
    print(f"[INFO] Results saved → {SAVE_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    steering_evaluation()
