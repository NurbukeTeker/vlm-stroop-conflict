# 🧠 CLIP Pseudoword Stroop Analysis

This experiment investigates how CLIP responds to **pseudoword Stroop stimuli** —  
nonsense words that resemble color names (e.g., *"blim", "gron"*) rendered in different ink colors.

The goal is to test whether CLIP’s language–vision alignment relies on **semantic understanding** (word meaning) or purely **visual color cues** when presented with *non-semantic text*.

---

## 🎯 Experimental Overview

Each pseudoword was generated for 10 color categories:

red, blue, green, yellow, orange, purple, brown, pink, gray, black


For every color set:
- **5 congruent images** (pseudoword rendered in its matching color)
- **45 incongruent images** (pseudoword rendered in other colors)

This yields **50 images per color**, 500 total across the dataset.

CLIP (ViT-B/32) was evaluated zero-shot with the prompt:
> “The word is written in {color} font.”

Predictions were classified as:
- **Congruent Correct** → model predicted the correct color when text and ink matched  
- **Color Correct** → model correctly recognized the ink color in incongruent settings  
- **Text Biased** → model chose a color semantically linked to the pseudoword (rare, since non-semantic)  
- **Off-Target** → neither text nor ink color matched prediction  

---

## 📊 Results per Color

| Color | Congruent ✓ | Color ✓ | Text Bias | Off-Target | Total | Ink Accuracy (%) |
|:------|-------------:|---------:|-----------:|------------:|-------:|-----------------:|
| black | 5 | 43 | 0 | 2 | 50 | **86.0** |
| blue | 4 | 40 | 0 | 5 | 50 | **80.0** |
| brown | 5 | 34 | 7 | 4 | 50 | **68.0** |
| gray | 5 | 33 | 10 | 2 | 50 | **66.0** |
| green | 5 | 35 | 0 | 10 | 50 | **70.0** |
| orange | 5 | 43 | 1 | 1 | 50 | **86.0** |
| pink | 5 | 39 | 3 | 3 | 50 | **78.0** |
| purple | 5 | 43 | 0 | 2 | 50 | **86.0** |
| red | 5 | 41 | 0 | 4 | 50 | **82.0** |
| yellow | 5 | 38 | 7 | 0 | 50 | **76.0** |

---

## 📈 Combined Summary

| Metric | Mean Across Colors |
|:-------|:------------------:|
| Congruent Correct | **5.0 / 50** |
| Color Correct | **38.9 / 50 (77.8%)** |
| Text Bias | **2.8 / 50 (5.6%)** |
| Off-Target | **3.3 / 50 (6.6%)** |

**Overall Ink Recognition Accuracy ≈ 78 %**

---

## 🧩 Interpretation

- CLIP identifies **ink color** accurately even when text lacks semantic meaning —  
  indicating strong *visual grounding* independent of language.
- Minimal “text bias” (< 6 %) confirms that pseudowords do **not** activate semantic priors.
- High congruent performance (always near 100 %) suggests CLIP still recognizes *visual color consistency*.
- The slight variance across hues (lower for gray/brown) aligns with human perceptual difficulty — darker or less saturated colors reduce model certainty.

---

## 🧠 Conclusion

The pseudoword experiment demonstrates that:
- CLIP’s **visual modality** remains dominant when semantic information is absent.  
- Its predictions depend primarily on **ink color perception**, not pseudo-lexical cues.  
- This contrasts with true word Stroop tests, where **semantic bias** strongly overrides visual features.

---

## 📂 Output Files

behavioural_analysis/pseudoword_analysis/results/
├── clip_predictions_<color>.csv # Raw CLIP predictions per color
├── summarized_predictions/<color>_summary.csv
└── merged_clip_prediction_summary.csv # Combined results table (above)