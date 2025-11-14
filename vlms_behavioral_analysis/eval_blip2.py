import os
import argparse
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from vlm_shared import (
    PROMPTS, load_stroop_images, parse_filename,
    normalize_text, classify_output, save_results_csv
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "Salesforce/blip2-flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODEL (Windows-safe)
# ==========================================================

def load_blip2():

    print("Loading BLIP-2 (FlanT5-XL)...")

    # Avoid tokenizer & symlink warnings on Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    processor = Blip2Processor.from_pretrained(
        MODEL_ID,
        use_fast=False   # ★ critical for Windows
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()
    return processor, model


# ==========================================================
# INFERENCE
# ==========================================================

def generate_answer(processor, model, image, prompt):
    """
    BLIP-2: image + prompt → causal LM generate
    """

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False
        )

    answer = processor.decode(generated_ids[0], skip_special_tokens=True)
    return answer.strip()


# ==========================================================
# MAIN EVALUATION
# ==========================================================

def main(prompt_key):

    if prompt_key not in PROMPTS:
        raise ValueError(f"Invalid prompt key={prompt_key}. Choose: short / strong / mc")

    prompt = PROMPTS[prompt_key]
    print(f"\nRunning BLIP-2 with prompt = {prompt_key}")
    print(f"Instruction:\n{prompt}\n")

    processor, model = load_blip2()
    image_paths = load_stroop_images()

    rows = []

    for path in image_paths:
        fname = os.path.basename(path)
        word, ink, condition = parse_filename(fname)

        if word is None:
            continue

        img = Image.open(path).convert("RGB")
        raw_answer = generate_answer(processor, model, img, prompt)

        # classify
        label = classify_output(raw_answer, word, ink)

        rows.append([
            fname, word, ink, condition,
            raw_answer, label
        ])

    out_file = f"results_blip2_{prompt_key}.csv"
    save_results_csv(rows, out_file)

    print(f"\nSaved → {out_file}")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="short",
                        choices=["short", "strong", "mc"])
    args = parser.parse_args()

    main(args.prompt)
