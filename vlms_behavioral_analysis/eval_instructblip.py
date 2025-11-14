import os
import argparse
import torch
from PIL import Image
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)
from vlm_shared import (
    PROMPTS, load_stroop_images, parse_filename,
    normalize_text, classify_output, save_results_csv
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "Salesforce/instructblip-vicuna-7b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODEL (Windows-safe fix)
# ==========================================================

def load_instructblip():

    print("Loading InstructBLIP model...")

    # ---- Windows tokenizer fix ----
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_NO_FAST_INIT"] = "1"

    # Force slow tokenizer → fixes Windows crash
    processor = InstructBlipProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False       # ⭐ critical fix ⭐
    )

    # Use fp16 if GPU available
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()
    return processor, model


# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_answer(processor, model, image, prompt):

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False
        )

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output.strip()


# ==========================================================
# MAIN
# ==========================================================

def main(prompt_key):

    if prompt_key not in PROMPTS:
        raise ValueError(f"Invalid prompt key={prompt_key}. Choose: short, strong, mc")

    prompt = PROMPTS[prompt_key]
    print(f"\nRunning InstructBLIP with PROMPT = {prompt_key}")
    print(f"Instruction:\n{prompt}\n")

    processor, model = load_instructblip()
    image_paths = load_stroop_images()

    rows = []

    for path in image_paths:
        fname = os.path.basename(path)
        word, ink, condition = parse_filename(fname)

        if word is None:
            continue

        img = Image.open(path).convert("RGB")
        raw_answer = generate_answer(processor, model, img, prompt)

        # Normalize + classify
        label = classify_output(raw_answer, word, ink)

        rows.append([
            fname, word, ink, condition, raw_answer, label
        ])

    out_file = f"results_instructblip_{prompt_key}.csv"
    save_results_csv(rows, out_file)
    print(f"\nSaved → {out_file}")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="short",
                        help="short / strong / mc")
    args = parser.parse_args()

    main(args.prompt)
