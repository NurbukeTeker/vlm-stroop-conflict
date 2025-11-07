# CLIP Visual Manipulation — Boldness Analysis

This experiment evaluates how **text boldness** (light, normal, bold, narrow) affects CLIP’s behavior in **Stroop-style color–word conflict** settings, while keeping the **font size fixed at 90 pt**.  
The goal is to determine whether CLIP’s predictions rely more on the **text meaning** or on the **visual ink color** when the same word is rendered with different stroke weights.

---

## 🎯 Experiment Overview

Each image contains a **color word** (e.g., “RED”) displayed in a **colored font** (e.g., blue).  
For every image, CLIP is evaluated with the prompt:

> “The word is written in {color} font.”

Predictions are then compared to:
- the **word meaning** (`word_text`) — e.g., *“RED”*  
- the **font color** (`font_color`) — e.g., *“blue”*

A correct **word prediction** indicates that CLIP focused on the *semantic meaning* of the word.  
A correct **color prediction** indicates that CLIP relied more on *visual appearance*.

---

## 🧩 Dataset Structure

Each boldness level corresponds to a separate folder containing 100 Stroop-style images per word:

behavioural_analysis/
└── visual_manipulations/
└── boldness_manipulation/
├── stroop_images_bold_light/
├── stroop_images_bold_normal/
├── stroop_images_bold_bold/
├── stroop_images_bold_narrow/
├── clip_visual_boldness.py
└── results/


Each subfolder contains images named in the format:

<word>/<word>as<color>.png

For example:
red/red_as_blue.png
blue/blue_as_red.png


---

## ⚙️ Running the Script

Run the following command from the project root:

```bash
python behavioural_analysis/visual_manipulations/boldness_manipulation/clip_visual_boldness.py

The script:

Loads the CLIP (ViT-B/32) model.

Iterates through all four boldness sets.

Computes accuracy with respect to both text and color cues.

Saves per-style CSVs and an aggregated summary to the results/ folder.

📊 Results Summary
Style	# Images	Text Accuracy (%)	Color Accuracy (%)	Interpretation
Light	100	89.00	21.00	Focus on text (Stroop effect)
Normal	100	88.00	22.00	Focus on text (Stroop effect)
Bold	100	89.00	21.00	Focus on text (Stroop effect)
Narrow	100	97.00	13.00	Focus on text (Stroop effect)

✅ Results saved to:
behavioural_analysis/visual_manipulations/boldness_manipulation/results/
├── clip_stroop_results_bold_light.csv
├── clip_stroop_results_bold_normal.csv
├── clip_stroop_results_bold_bold.csv
├── clip_stroop_results_bold_narrow.csv
└── clip_visual_boldness_summary.csv

Conclusion

Changing boldness of the text does not significantly shift CLIP’s bias—its perception remains dominated by the textual meaning of the word.
This finding reinforces the notion that semantic features override visual ones in CLIP’s joint embedding space when text and color conflict.