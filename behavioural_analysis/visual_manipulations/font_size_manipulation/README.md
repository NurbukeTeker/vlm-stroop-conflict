# 🧠 CLIP Visual Manipulation — Font Size Analysis

This experiment evaluates the **visual robustness** of the CLIP model across different **font sizes** (48 pt, 72 pt, 90 pt, 108 pt) using **Stroop-style color–word conflict stimuli**.  
The goal is to assess whether CLIP’s zero-shot predictions are more influenced by the **written text meaning** or the **visual ink color** when the text size changes.

---

## 🎯 Experiment Overview

Each image displays a **color word** (e.g., “RED”) written in a **colored font** (e.g., blue, green, etc.).  
For every image, CLIP is asked to classify the image with the prompt:

> “The word is written in {color} font.”

Predictions are then compared to:
- the **text meaning** (`word_text`) — “RED”
- the **font color** (`font_color`) — e.g. “blue”

A correct **word prediction** indicates semantic (textual) dominance,  
while a correct **color prediction** indicates perceptual (visual) dominance.

---

## 🧩 Dataset Structure

Each font size variation is stored in a separate folder:

behavioural_analysis/
└── visual_manipulations/
└── font_size_manipulation/
├── stroop_images_base_48/
├── stroop_images_base_72/
├── stroop_images_base_90/
└── stroop_images_base_108/


Each subfolder contains 100 images per condition, organized by word class  
(e.g., `red/red_as_blue.png`, `blue/blue_as_red.png`, etc.).

---

## ⚙️ Running the Script

Execute the experiment using:

```bash
python behavioural_analysis/visual_manipulations/font_size_manipulation/clip_visual_fontsize.py


The script will:

Load CLIP (openai/clip-vit-base-patch32)

Evaluate all four font-size datasets

Save per-size results and a global summary table under
behavioural_analysis/results/

📊 Results Summary
Font Size	# Images	Text Accuracy (%)	Color Accuracy (%)	Interpretation
48 pt	100	97.00	13.00	🧠 Focus: text (Stroop effect)
72 pt	100	95.00	15.00	🧠 Focus: text (Stroop effect)
90 pt	100	89.00	21.00	🧠 Focus: text (Stroop effect)
108 pt	101	77.23	32.67	🧠 Focus: text (Stroop effect)

behavioural_analysis/results/
├── clip_stroop_results_48pt.csv
├── clip_stroop_results_72pt.csv
├── clip_stroop_results_90pt.csv
├── clip_stroop_results_108pt.csv
└── clip_visual_fontsize_summary.csv

Interpretation

CLIP maintains a strong text-dominant bias across all font sizes.

As font size increases, color accuracy rises slightly, indicating that larger text makes the color cue more visually salient.

Nonetheless, even at 108 pt, textual meaning still dominates the model’s zero-shot decisions — consistent with a semantic Stroop effect.
