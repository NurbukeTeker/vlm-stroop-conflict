import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlm_shared import (
    PROMPTS, load_stroop_images, parse_filename,
    classify_output, save_results_csv
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "liuhaotian/llava-v1.6-vicuna-7b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODEL (Windows-safe)
# ==========================================================

def load_llava():
    print("Loading LLaVA-1.6 Vicuna-7B...")

    # Windows fixes
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False          # ★ critical for Windows (Vicuna tokenizer)
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    model.eval()
    return processor, model


# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_answer(processor, model, image, instruction):
    """
    LLaVA inference = VQA style
    Format: <image> + text prompt
    """

    prompt = f"USER: <image>\n{instruction}\nASSISTANT:"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


# ==========================================================
# MAIN
# ==========================================================

def main(prompt_key):

    if prompt_key not in PROMPTS:
        raise ValueError(f"Invalid prompt key={prompt_key}. Use: short / strong / mc")

    instruction = PROMPTS[prompt_key]
    print(f"\nRunning LLaVA with prompt = {prompt_key}")
    print(f"Instruction:\n{instruction}\n")

    processor, model = load_llava()
    image_paths = load_stroop_images()

    rows = []

    for path in image_paths:
        fname = os.path.basename(path)
        word, ink, condition = parse_filename(fname)
        if word is None:
            continue

        img = Image.open(path).convert("RGB")
        raw_answer = generate_answer(processor, model, img, instruction)

        # classify raw output
        label = classify_output(raw_answer, word, ink)

        rows.append([
            fname, word, ink, condition,
            raw_answer, label
        ])

    outfile = f"results_llava_{prompt_key}.csv"
    save_results_csv(rows, outfile)

    print(f"\nSaved → {outfile}")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="short",
                        choices=["short", "strong", "mc"])
    args = parser.parse_args()

    main(args.prompt)
