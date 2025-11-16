import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ============================================================
# PATHS
# ============================================================
STEERING_RESULTS = "saved/results/steering_cosine.json"
EMBED_ROOT = "saved/embeddings/congruent"
CHUNK_COLOR_ROOT = "saved/chunks/color"
CHUNK_TEXT_ROOT = "saved/chunks/text"

SAVE_PATH = "saved/results/behavioral_eval.json"

TARGET_COLORS = ["red","blue","green","yellow","orange","purple","pink","brown","gray","black"]


# ============================================================
# LOAD MODEL
# ============================================================
def load_qwen():
    print("[INFO] Loading Qwen2-VL-7B-Instruct for behavioral evaluation...")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


# ============================================================
# LOAD ORIGINAL EMBEDDING (BEFORE STEERING)
# ============================================================
def load_original_embedding(filename, layer_idx):
    path = os.path.join(
        EMBED_ROOT,
        filename.split("_as_")[0],  # word
        f"layer{layer_idx}",
        filename
    )
    return torch.load(path, map_location="cpu").squeeze(0)


# ============================================================
# LOAD CHUNKS
# ============================================================
def load_chunks(root_dir):
    layers = {}
    for f in os.listdir(root_dir):
        if f.endswith("_all.pt"):
            layer_idx = int(f.replace("layer","").replace("_all.pt",""))
            layers[layer_idx] = torch.load(os.path.join(root_dir, f), map_location="cpu")
    return layers


# ============================================================
# APPLY STEERING
# ============================================================
def apply_chunk(original_vec, chunk_vec):
    new_vec = original_vec + chunk_vec
    return new_vec / new_vec.norm()


# ============================================================
# RUN QWEN ON STEERED EMBEDDING
# ============================================================
def run_qwen_with_steered_embedding(model, processor, steered_emb):
    """
    Patch Qwen’s visual encoder output so that the model
    uses the steered embedding directly.
    """
    B = 1

    def fake_vision_forward(*args, **kwargs):
        # Expand to B, SeqLen=1
        return steered_emb.unsqueeze(0).unsqueeze(1).to(model.device)

    # Monkey patch
    original_forward = model.vision_tower.forward
    model.vision_tower.forward = fake_vision_forward

    # Run inference
    prompt = "What is the ink color of the text?"
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    answer = processor.decode(out[0], skip_special_tokens=True)

    # Remove patch
    model.vision_tower.forward = original_forward

    return answer.lower().strip()


# ============================================================
# PARSE COLOR FROM ANSWER
# ============================================================
def normalize_color(text):
    for c in TARGET_COLORS:
        if c in text.lower():
            return c
    return "unknown"


# ============================================================
# MAIN BEHAVIORAL EVAL PROCESS
# ============================================================
def run_behavioral_evaluation():
    print("[INFO] Loading steering cosine results...")
    with open(STEERING_RESULTS, "r") as f:
        steering_data = json.load(f)

    model, processor = load_qwen()

    print("[INFO] Loading chunks...")
    color_chunks = load_chunks(CHUNK_COLOR_ROOT)
    text_chunks  = load_chunks(CHUNK_TEXT_ROOT)

    results = []

    print("[INFO] Starting BEHAVIORAL steering evaluation...")

    for item in steering_data:
        filename  = item["filename"].replace(".pt",".png")
        layer_idx = item["layer"]
        src_color = item["source_color"]
        src_word  = item["source_word"]
        target    = item["target"]

        # Load original embedding
        orig_emb = load_original_embedding(item["filename"], layer_idx)

        # color chunk
        if layer_idx in color_chunks:
            col_map = color_chunks[layer_idx]
            if src_color in col_map and target in col_map:
                c_chunk = col_map[target] - col_map[src_color]
                c_chunk = c_chunk / c_chunk.norm()
                steered_color = apply_chunk(orig_emb, c_chunk)

                answer = run_qwen_with_steered_embedding(model, processor, steered_color)
                pred = normalize_color(answer)
            else:
                pred = "unknown"
        else:
            pred = "unknown"

        results.append({
            "filename": filename,
            "layer": layer_idx,
            "source_color": src_color,
            "source_word": src_word,
            "target": target,
            "behavior_prediction": pred
        })

    print("[INFO] Saving behavioral results...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("[INFO] DONE. Behavioral evaluation saved →", SAVE_PATH)


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    run_behavioral_evaluation()
