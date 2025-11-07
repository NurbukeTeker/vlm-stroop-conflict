# 🎨 CLIP Contrast Manipulation Analysis

This section investigates how **background contrast** affects CLIP’s attention balance between **textual meaning** and **visual ink color** in Stroop-style stimuli.

Two conditions are evaluated:
- **Congruent** → word meaning and font color are the same (e.g., “RED” written in red).  
- **Incongruent** → word meaning and font color conflict (e.g., “RED” written in blue).

---

## 🧩 Dataset Overview

All images were generated using a custom contrast pipeline:

stroop_contrast_variants/ # Congruent (word=color)
stroop_incongruent_contrast_variants/ # Incongruent (word≠color)


Each word (10 color categories) was rendered under **four contrast levels**:
`high`, `medium`, `low`, `same`.

---

## 📊 CLIP Accuracy on Congruent Contrast Stimuli

| Contrast Level | Word Accuracy (%) | Color Accuracy (%) | Images | Observation |
|----------------|------------------:|-------------------:|---------:|--------------|
| High | **100.0** | **100.0** | 40 | Both cues identical — perfect alignment |
| Medium | **100.0** | **100.0** | 40 | No Stroop conflict, invariant performance |
| Low | **100.0** | **100.0** | 40 | Same accuracy even with reduced contrast |
| Same | **100.0** | **100.0** | 40 | Contrast has no effect when modalities align |

**Summary:**  
In congruent settings, CLIP performs perfectly across all contrast levels.  
Because text and color convey the same signal, we cannot isolate which modality the model relied on.

---

## ⚖️ CLIP Accuracy on Incongruent Contrast Stimuli

| Contrast Level | Word Accuracy (%) | Color Accuracy (%) | Images | Interpretation |
|----------------|------------------:|-------------------:|---------:|----------------|
| High | **87.8** | **12.2** | 400 | Clear **text bias** — semantic content dominates |
| Medium | **83.3** | **16.7** | 400 | Slight drop in text reliability, still text-driven |
| Low | **67.8** | **32.2** | 400 | Visual ink color starts influencing predictions |
| Same | **0.0** | **100.0** | 400 | Text unreadable, CLIP fully relies on color cue |

---

## 🧠 Interpretation and Insights

1. **Stroop Effect and Text Dominance**  
   Under high contrast, CLIP overwhelmingly prioritizes the written word (~88% text vs. 12% color), confirming its semantic bias.

2. **Color Influence Increases with Lower Contrast**  
   As background contrast decreases, text saliency weakens. The model’s predictions begin to reflect ink color more strongly.

3. **Same Contrast Condition (Text = Background)**  
   When the background and text share the same color, the word becomes nearly invisible.  
   CLIP switches entirely to the color modality, predicting based solely on the visual tone rather than the word meaning.

---

## 📈 Combined Contrast–Bias Trend

| Contrast | Word Accuracy | Color Accuracy | Dominant Modality |
|-----------|----------------|----------------|-------------------|
| High | 88 % | 12 % | Text |
| Medium | 83 % | 17 % | Text |
| Low | 68 % | 32 % | Mixed |
| Same | 0 % | 100 % | Color |

---

## 🧭 Conclusion

- CLIP’s semantic encoding remains **robust to contrast variations** as long as the text remains legible.  
- When **contrast diminishes**, semantic reliability declines, and the model begins to rely on **visual color cues** instead.  
- This gradient shift reveals a **contrast-dependent Stroop effect**, where perception gradually transitions from **reading** to **seeing**.

---

**Results stored in:**  

behavioural_analysis/visual_manipulations/contrast_manipulation/results/
├── clip_stroop_congruent_contrast_summary.csv
├── clip_stroop_incongruent_contrast_summary.csv
└── clip_contrast_combined_summary.csv
