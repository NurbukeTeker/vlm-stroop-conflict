VLMRepo — Stroop-Style Multimodal Conflict in Vision–Language Models
Overview

This repository reproduces the experiments and analyses presented in:

"What is the Color of RED? Vision–Language Models Prefer to Read Rather Than See"
Under review at ICLR 2026
Nurbüke Teker, Technical University of Munich (M.Sc. Informatics Thesis, 2025)

The project investigates how Vision–Language Models (VLMs) resolve conflicting cues between written words and their ink colors — adapting the classical Stroop paradigm from psychology.
Our findings reveal a striking textual dominance: when the word and its ink color conflict, VLMs “read” the image rather than “see” it.

Research Goals

Adapt the Stroop test to multimodal models (CLIP, SigLIP-2, LLaVA, BLIP-2, Kosmos-2, Qwen-VL, etc.)

Quantify word vs. color bias behaviorally and representationally

Analyze embedding-space geometry via RDMs and UMAP

Apply subpopulation-based latent interventions (word vs color chunks)

Demonstrate how representation saliency → steerability

Repository Structure

VLMRepo/
│
├── datasets/
│   └── stroop_images/          # 100 balanced Stroop stimuli (10 colors × 10 words)
│
├── behavioural_analysis/
│   ├── clip_stroop_test.py     # CLIP & SigLIP-2 behavioural evaluation
│   ├── generative_vlm_eval.py  # LLaVA, BLIP-2, Kosmos-2, etc.
│   └── results/                # Aggregated CSVs and plots
│
├── embedding_analysis/
│   ├── rdm_analysis.py         # Representational Dissimilarity Matrices
│   ├── umap_projection.py      # Low-dimensional visualization
│   └── plots/                  # Saved RDM & UMAP figures
│
├── interventions/
│   ├── utils_intervention.py   # Model loading & embedding utilities
│   ├── llava_color_intervention.py  # Latent steering experiments
│   └── results/                # Intervention success heatmaps
│
├── notebooks/
│   └── vlm_stroop_summary.ipynb  # End-to-end demonstration notebook
│
├── figures/
│   ├── fig1_stroop_scheme.png
│   ├── fig2_behavioral_results.png
│   ├── fig4_rdm_umap.png
│   ├── fig5_intervention_pipeline.png
│   └── fig6_intervention_heatmaps.png
│
└── README.md  # (this file)


Methodology
1. Stroop Dataset

100 images with 10 color classes (red, blue, green, yellow, orange, purple, pink, brown, gray, black) rendered both congruently and incongruently.
Each file follows the format:

<WORD>_<INK>.png   →  e.g.,  BLUE_RED.png

2. Behavioural Analysis

Contrastive models (CLIP, SigLIP-2):

Prompts:

Word-oriented → “The text says RED.”

Ink-oriented → “The text is written in red color.”

Accuracy measured via cosine similarity between image and prompt embeddings.

Model	Word = Ink	Word Acc.	Ink Acc.
CLIP	100.0	97.8	20.0
SigLIP-2	100.0	100.0	5.6

Generative models (LLaVA, BLIP-2, Kosmos-2, GIT, Qwen2-VL):
Despite being asked “What color is the word in this image?”, models overwhelmingly respond with the word instead of the ink color.
LLaVA is the only model that sometimes prefers ink (≈ 54% Ink Match).

3. Representation Analysis

RDMs: show that adding word cues reshapes embedding space more sharply than adding color.

UMAP: image embeddings cluster by written word, not ink color.

<p align="center"><img src="figures/fig4_rdm_umap.png" width="600"></p>
4. Latent Interventions

We implement subpopulation-based steering (Wu et al., 2025):
Identify stable concept-encoding dimensions, compute mean chunk vectors per concept, and edit embeddings as:

$$ 
E' = E - \mu_{\text{src}} + \mu_{\text{tgt}}
$$

	​

Intervention Type	Success (%)	Mean Δ (sim shift)
Ink-Color	36.7	−0.0017
Word	100.0	+0.0934
Combined	100.0	+0.1172

Word and combined directions are modular and steerable; color directions are weak and collinear.

<p align="center"><img src="figures/fig6_intervention_heatmaps.png" width="600"></p>

Key Findings
Aspect	Observation
Behavioral	VLMs “read” rather than “see” under conflict.
Representation	Word cues form stronger, separable clusters.
Intervention	Word chunks steer embeddings reliably; color chunks fail.
Generalization	Text bias persists across fonts, contrasts, and pseudowords.
⚙️ Installation & Usage
git clone https://github.com/<yourname>/VLMRepo.git
cd VLMRepo
pip install -r requirements.txt


To reproduce CLIP and LLaVA experiments:

python behavioural_analysis/clip_stroop_test.py
python interventions/llava_color_intervention.py

Citation

If you use this repository or dataset, please cite:

@article{teker2026vlmstrop,
  title     = {What is the Color of RED? Vision–Language Models Prefer to Read Rather Than See},
  author    = {Nurbüke Teker},
  journal   = {Under review at ICLR},
  year      = {2026}
}

Acknowledgments

This repository builds upon the master’s thesis work conducted at
Technical University of Munich (TUM) under supervision of Prof. Evrim Anıl Evirgen,
and integrates official implementations of CLIP, SigLIP-2, LLaVA, and BLIP-2.

Next Steps

Extend latent intervention framework to newer VLMs (e.g., InternVL-2.5, Qwen-VL-2.5)

Explore naturalistic conflicts (texture, material, shape)

Develop color-robust fine-tuning strategies for multimodal bias mitigation