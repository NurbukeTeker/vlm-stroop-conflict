import os
import torch
from typing import Dict, List

# ============================================================
# PATHS
# ============================================================
EMBED_ROOT = "saved/embeddings"
SAVE_ROOT = "saved/chunks"

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# LOAD ALL EMBEDDINGS OF A GIVEN CLASS & LAYER
# ============================================================
def load_embeddings_for_class(base_dir: str, label: str) -> Dict[int, List[torch.Tensor]]:
    """
    Reads saved embeddings:
    base_dir/{label}/layerX/*.pt
    Returns dict: {layer_idx: [emb1, emb2, ...]}
    """
    class_dir = os.path.join(base_dir, label)
    if not os.path.isdir(class_dir):
        return {}

    layers = {}
    for layer_name in os.listdir(class_dir):
        if not layer_name.startswith("layer"):
            continue

        layer_idx = int(layer_name.replace("layer", ""))
        layer_dir = os.path.join(class_dir, layer_name)
        if not os.path.isdir(layer_dir):
            continue

        embeddings = []
        for f in os.listdir(layer_dir):
            if f.endswith(".pt"):
                emb = torch.load(os.path.join(layer_dir, f), map_location="cpu").squeeze(0)
                embeddings.append(emb)

        if len(embeddings) > 0:
            layers[layer_idx] = embeddings

    return layers


# ============================================================
# CHUNK COMPUTATION
# ============================================================
def compute_chunk(emb_list: List[torch.Tensor]):
    """
    Average + normalize.
    """
    stacked = torch.stack(emb_list, dim=0)
    mean = stacked.mean(dim=0)
    mean = mean / mean.norm()
    return mean


# ============================================================
# SAVE CHUNK
# ============================================================
def save_chunk(tensor: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


# ============================================================
# MAIN: BUILD CHUNKS FOR COLOR-ONLY AND TEXT-ONLY
# ============================================================
def process_modality(modality: str):
    """
    modality ∈ {"color_only", "text_only"}
    Produces:
       saved/chunks/{modality_type}/layerX/{label}.pt
       saved/chunks/{modality_type}/layerX_all.pt
    """
    print(f"[INFO] Processing modality: {modality}")

    modality_dir = os.path.join(EMBED_ROOT, modality)
    if not os.path.isdir(modality_dir):
        raise RuntimeError(f"[ERROR] Embedding directory not found: {modality_dir}")

    labels = sorted([d for d in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, d))])
    print(f"[INFO] Found labels: {labels}")

    # To collect aggregated per-layer chunks:
    layerwise_dict = {}

    for label in labels:
        print(f"[INFO] → Processing label: {label}")

        class_layers = load_embeddings_for_class(modality_dir, label)

        for layer_idx, emb_list in class_layers.items():
            chunk = compute_chunk(emb_list)

            # Save per-label chunk
            save_path = os.path.join(SAVE_ROOT, modality.replace("_only", ""), f"layer{layer_idx}", f"{label}.pt")
            save_chunk(chunk, save_path)

            # Aggregate
            layerwise_dict.setdefault(layer_idx, {})[label] = chunk

    # Save layer-wise combined dict
    for layer_idx, dict_layer in layerwise_dict.items():
        save_path = os.path.join(SAVE_ROOT, modality.replace("_only", ""), f"layer{layer_idx}_all.pt")
        torch.save(dict_layer, save_path)

    print(f"[INFO] Completed modality: {modality}")


# ============================================================
# RUN
# ============================================================
def main():
    # COLOR CHUNKS
    process_modality("color_only")

    # TEXT CHUNKS
    process_modality("text_only")

    print("[INFO] All chunks built and saved.")


if __name__ == "__main__":
    main()
