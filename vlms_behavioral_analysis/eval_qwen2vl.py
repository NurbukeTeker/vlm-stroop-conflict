import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from vlm_shared import (
    PROMPTS, load_stroop_images, parse_filename,
    classify_output, save_results_csv
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ==========================================================
# LOAD MODEL (Windows + Qwen2-VL safe)
# ==========================================================

def load_qwen2vl():
    print("Loading Qwen2-VL-7B-Instruct...")

    # Windows fixes
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Qwen2-VL requires remote code + non-fast tokenizer
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    model.eval()
    return processor, model


# ==========================================================
# GENERATE ANSWER (Qwen2-VL chat format)
# ==========================================================

def generate_answer(processor, model, image, instruction):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    # Build multimodal prompt
    inputs = processor.apply_chat_template(
        messages,
        images=[image],
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )

    output_text = processor.decode(output_ids[0], skip_special_tokens=True)
    return output_text.strip()


# ==========================================================
# MAIN
# ==========================================================

def main(prompt_key):

    if prompt_key not in PROMPTS:
        raise ValueError("Invalid prompt key. Use: short / strong / mc")

    instruction = PROMPTS[prompt_key]
    print(f"\nRunning Qwen2-VL with prompt = {prompt_key}")
    print(f"Instruction:\n{instruction}\n")

    processor, model = load_qwen2vl()
    image_paths = load_stroop_images()

    rows = []

    for path in image_paths:
        fname = os.path.basename(path)
        word, ink, condition = parse_filename(fname)
        if word is None:
            continue

        img = Image.open(path).convert("RGB")
        raw_answer = generate_answer(processor, model, img, instruction)
        label = classify_output(raw_answer, word, ink)

        rows.append([fname, word, ink, condition, raw_answer, label])

    outname = f"results_qwen2vl_{prompt_key}.csv"
    save_results_csv(rows, outname)

    print(f"\nSaved → {outname}")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="short",
                        choices=["short", "strong", "mc"])
    args = parser.parse_args()

    main(args.prompt)
