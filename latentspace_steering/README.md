# Qwen Latent Steering Pipeline — Step-by-Step Summary

Below is a clear summary of everything we have implemented so far on the Qwen2-VL latent steering analysis, including inputs, outputs, models used, and the purpose of each step.

## 1. Goal of the Qwen Experiments

We aim to test whether latent interventions (color-chunks and word-chunks)—computed from the Stroop dataset—can steer Qwen2-VL’s internal visual representations and consequently change its color predictions.

This mirrors what was previously done for CLIP and SigLIP, but here we apply it inside a generative VLM (Qwen2-VL).

## 2. Step 1 — Extract Vision Encoder Embeddings

Script: 1_extract_embeddings.py
Model: Qwen/Qwen2-VL-7B-Instruct (vision tower only)
Input: every Stroop image in 3 subsets:

color_only/

text_only/

congruent/

What we do:

Register forward hooks over all 32 Qwen vision layers

For every image, capture the CLS embedding for each layer

Save results as:

saved/embeddings/{subset}/{label}/layer{L}/{image}.pt


Purpose:
This gives us the full layer-wise latent representation of the model for colors, words, and congruent conditions.

## 3. Step 2 — Subpopulation Chunk Construction

Script: 2_build_chunks_subpopulation.py
Input: embeddings from Step 1
Method: Subpopulation-based chunking using K-Means clustering

For each color (red, blue, …) and for each layer L:

Take all embeddings belonging to that color

Run K-Means to form K=2 subpopulations

Save each cluster centroid (normalized)

Saved under:

saved/chunks_subpop/color/layer{L}_all.pt
saved/chunks_subpop/text/layer{L}_all.pt


Each file maps:

{
   "red": vector(1280),
   "blue": vector(1280),
   ...
}


Purpose:
These chunks capture meaningful latent neighborhoods instead of mean-based templates.
They represent directional features that encode color or word semantics in Qwen’s latent space.

## 4. Step 3 — Latent Steering Evaluation (Cosine-Level)

Script: 3_run_steering_eval.py
Input:

Congruent embeddings

Subpopulation chunks for color and text

What we compute:

For each congruent image like "blue_as_red"
and for each target color t ≠ source:

(a) Color steering vector:
Δ_color = chunk_color[target] – chunk_color[source_color]

(b) Word steering vector:
Δ_text = chunk_text[target] – chunk_text[source_word]

(c) Combined steering:
Δ_combined = -chunk[source_color] -chunk[source_word]
             +chunk[target_color] +chunk[target_word]


Then we compute before vs. after cosine similarity to target color embeddings:

cos(original, text_emb[target])
cos(steered, text_emb[target])


Saved as:

saved/results/steering_subpop.json


Purpose:
This tests whether steering directions move the image embedding closer to the target color concept inside Qwen’s latent space.

## 5. Step 4 — Behavioral Steering Evaluation (Optional)

We attempted to evaluate whether latent interventions change the actual output text of Qwen (e.g., “What is the ink color?”).

Pipeline:

Monkey-patch Qwen’s vision tower to return our steered embedding instead of real image features

Feed a fixed prompt:

“What is the ink color of the text?”

Collect predicted color word

Store results in:

saved/results/behavioral_eval.json


Note:
Large Qwen models (7B) cannot load on CPU; we will run behavioral tests on GPU (LRZ cluster) or using 2B variant.

Purpose:
Checks whether steering not only shifts internal representations but also changes the model’s output answer — this is the behavioral effect.

## 6. Directory Summary

Final output directories:

saved/
   embeddings/       ← raw layerwise embeddings (CLS)
   chunks_subpop/    ← subpopulation-based color/word chunks
   results/
       steering_subpop.json   ← cosine-level steering effects
       behavioral_eval.json   ← behavioral effects (optional)

## 7. What This Enables

With this pipeline we can now:

Compare color vs. word steering strength across layers

Identify layers where color representations are most modifiable

Validate whether latent steering creates observable behavioral changes

Provide strong evidence about interpretable latent directions in Qwen2-VL