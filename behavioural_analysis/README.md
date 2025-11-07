# 🧠 CLIP Stroop-Style Behavioral Analysis

This directory contains behavioral evaluation scripts for reproducing **Stroop-style multimodal conflict experiments** on **CLIP**.  
The goal is to analyze whether CLIP prioritizes **textual meaning** or **visual color** when presented with conflicting cues.

---

## 📁 Dataset Overview

We use a **synthetic Stroop dataset** of **100 images**, systematically pairing 10 color words  
(`red, blue, green, yellow, orange, purple, brown, pink, gray, black`)  
with 10 corresponding ink colors.

Each image follows the filename pattern:

```text
<word>/<word>as<color>.png
```

### Example Structure


```text
stroop_images/
│
├── red/
│   ├── red_as_red.png
│   ├── red_as_blue.png
│   └── ...
│
└── blue/
    ├── blue_as_blue.png
    ├── blue_as_red.png
    └── ...

```


| Image File | Word | Ink Color | Congruency |
|-------------|-------|-----------|-------------|
| `red/red_as_red.png` | RED | red | ✅ Congruent |
| `blue/blue_as_green.png` | BLUE | green | ❌ Incongruent |

```text
Each color folder contains 10 PNG images, one for every possible color pairing (10×10 = 100 total images).
```


## 🧩 Experimental Goal

```text
The Stroop paradigm tests **modality dominance** in Vision-Language Models — whether the model “reads” or “sees” more strongly.
```

| Prompt Type | Example Prompt | Focus |
|--------------|----------------|--------|
| **Word-Oriented** | “The text says BLUE.” | Semantic meaning (word identity) |
| **Ink-Oriented** | “The text is written in blue color.” | Visual appearance (font color) |

```text
By comparing model accuracy under **congruent** (word = color)  
and **incongruent** (word ≠ color) stimuli,  
we reveal whether CLIP favors textual or visual information when these conflict.
```


## ⚙️ Running the Analysis

Make sure you have the environment set up (Python ≥ 3.10, CUDA optional):

```bash
conda activate vlm-stroop-conflict
pip install torch torchvision transformers pillow pandas tqdm scikit-learn
```


Then run any of the scripts below:

1️⃣ Word-Oriented Evaluation
python behavioural_analysis/clip_word_oriented_analysis.py


→ Uses prompts like “The text says BLUE.”

2️⃣ Ink-Oriented Evaluation
python behavioural_analysis/clip_ink_oriented_analysis.py


→ Uses prompts like “The text is written in blue color.”

3️⃣ Mixed (Prompt-Similarity) Evaluation
python behavioural_analysis/clip_behavioural_analysis.py


→ Compares both word- and color-oriented prompt embeddings on all 100 stimuli.

All results are automatically saved as:

behavioural_analysis/results/clip_stroop_results.csv



📊 Example Results

Typical results (reproduced using openai/clip-vit-base-patch32):


| Condition       | Prompt Type | Accuracy |
| --------------- | ----------- | -------- |
| **Congruent**   | Word        | 1.000    |
| **Congruent**   | Ink         | 1.000    |
| **Incongruent** | Word        | 1.000    |
| **Incongruent** | Ink         | 0.089    |





🧠 Interpretation

✅ Congruent cases: CLIP correctly recognizes both word and ink color.

❌ Incongruent cases: CLIP overwhelmingly follows the word rather than the color.

This reproduces the Stroop-style pattern reported in the thesis:

CLIP “reads” instead of “sees” — it prioritizes textual semantics even when instructed to focus on color.

🖼️ Example Visualization
Image	Color-Prompt	Text-Prompt	CLIP Prediction

	“The text is written in red color.” ✅	“The text says RED.” ✅	Both correct

	“The text is written in green color.” ❌	“The text says BLUE.” ✅	Word-dominant
📂 Output Files
File	Description
clip_word_oriented_analysis.py	Evaluates text-oriented prompts (“The text says X”)
clip_ink_oriented_analysis.py	Evaluates ink-oriented prompts (“The text is written in X color”)
clip_behavioural_analysis.py	Mixed evaluation comparing both modalities
results/clip_stroop_results.csv	Output CSV containing per-image predictions & accuracy
🧾 Citation

If you use this dataset or evaluation framework, please cite:

Teker, N. (2025). When VLMs Read Instead of See: Text Dominance in Multimodal Conflict.
Technical University of Munich, Department of Informatics.

🔍 Key Takeaway

Across all tests, CLIP consistently shows a strong text dominance bias:

When text and color conflict, it almost always “reads” the word.

Even with explicit color prompts, visual color recognition remains minimal (~9%).

This behavioral signature aligns perfectly with Stroop-style interference effects in cognitive psychology — revealing that CLIP, like humans, struggles to suppress its dominant modality (language) when faced with conflicting multimodal cues.










