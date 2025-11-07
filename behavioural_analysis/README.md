
| Image File | Word | Ink Color | Congruency |
|-------------|-------|-----------|-------------|
| `red_as_red.png` | RED | red | ✅ Congruent |
| `blue_as_green.png` | BLUE | green | ❌ Incongruent |

Each color folder contains 10 PNG images, one for every possible color pairing.

---

## 🧩 Experimental Goal

The Stroop paradigm tests **modality dominance** in Vision-Language Models — whether the model “reads” or “sees” more strongly.

| Prompt Type | Example Prompt | Focus |
|--------------|----------------|--------|
| **Word-Oriented** | “The text says BLUE.” | Semantic meaning (word identity) |
| **Ink-Oriented** | “The text is written in blue color.” | Visual appearance (font color) |

By comparing model accuracy under **congruent** (word = color)  
and **incongruent** (word ≠ color) stimuli,  
we reveal whether CLIP favors textual or visual information when these conflict.

---

## ⚙️ Running the Analysis

Make sure you have the environment set up (Python ≥ 3.10, CUDA optional):

```bash
conda activate vlm-stroop-conflict
pip install torch torchvision transformers pillow pandas tqdm scikit-learn
