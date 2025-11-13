Stroop-Style Multimodal Conflict Dataset Generator

This repository provides a controlled dataset generator for Stroop-style multimodal conflict images, designed to evaluate how Vision–Language Models (VLMs) such as CLIP, SigLIP, LLaVA, or Qwen-VL respond to word–color conflicts.

The generator applies variations in ink color, background color, tone, brightness, and saturation, with strict contrast filtering to maintain image quality and readability.

Final dataset size after filtering: ~2300–2600 images.

🎨 1. Color Vocabulary (10 Base Colors)

The generator uses the paper-standard RGB values shown below:

{
  "red":    (255, 0, 0),
  "blue":   (0, 0, 255),
  "green":  (0, 128, 0),
  "yellow": (255, 255, 0),
  "orange": (255, 165, 0),
  "purple": (128, 0, 128),
  "brown":  (139, 69, 19),
  "pink":   (255, 192, 203),
  "gray":   (128, 128, 128),
  "black":  (0, 0, 0)
}


These colors serve both as:

- the ink color used to render text, and

- the semantic word displayed in the image.

🎛 2. Tone Variations (5 Levels)

Each base ink color is expanded into five tone intensities using multiplicative scaling:

[0.6, 0.8, 1.0, 1.2, 1.4]


These produce:

- lighter tones

- pastel-like tones

- natural baseline

- saturated variants

- deeper tones

🌈 3. Background Variations (6 Pastel Tones)

Six soft, high-luminance pastel backgrounds are used to introduce variation while avoiding overwhelming the Stroop signal.

[
  (255, 255, 255),   # white
  (235, 235, 235),   # soft gray
  (245, 240, 225),   # beige / cream
  (250, 230, 235),   # soft pink
  (230, 245, 235),   # soft mint green
  (250, 250, 225)    # soft pale yellow
]


Background names used in filenames:

RGB	Name
(255,255,255)	white
(235,235,235)	softgray
(245,240,225)	beige
(250,230,235)	softpink
(230,245,235)	mint
(250,250,225)	softyellow
💡 4. Brightness & Saturation Augmentations

To model perceptual-level variation, each image uses:

Saturation levels:
[0.8, 1.0, 1.2]

Brightness levels:
[0.8, 1.0, 1.2]


These produce subtle but meaningful visual variations.

🔍 5. Quality Control — Contrast Filtering

Images are only kept if:

RGB_distance(ink_color, background_color) ≥ 40


# Stroop-Style Multimodal Conflict Dataset Generator

This repository provides a controlled dataset generator for Stroop-style multimodal conflict images, designed to evaluate how Vision–Language Models (VLMs) such as CLIP, SigLIP, LLaVA, or Qwen-VL respond to word–color conflicts.

The generator applies variations in ink color, background color, tone, brightness, and saturation, with strict contrast filtering to maintain image quality and readability.

Final dataset size after filtering: ~2300–2600 images.

## Color Vocabulary (10 Base Colors)

The generator uses the paper-standard RGB values shown below:

```json
{
  "red":    [255, 0, 0],
  "blue":   [0, 0, 255],
  "green":  [0, 128, 0],
  "yellow": [255, 255, 0],
  "orange": [255, 165, 0],
  "purple": [128, 0, 128],
  "brown":  [139, 69, 19],
  "pink":   [255, 192, 203],
  "gray":   [128, 128, 128],
  "black":  [0, 0, 0]
}
```

These colors serve both as:

- the ink color used to render text, and
- the semantic word displayed in the image.

## Tone Variations (5 Levels)

Each base ink color is expanded into five tone intensities using multiplicative scaling:

`[0.6, 0.8, 1.0, 1.2, 1.4]`

These produce:

- lighter tones
- pastel-like tones
- natural baseline
- saturated variants
- deeper tones

## Background Variations (6 Pastel Tones)

Six soft, high-luminance pastel backgrounds are used to introduce variation while avoiding overwhelming the Stroop signal.

```text
[
  (255, 255, 255),   # white
  (235, 235, 235),   # soft gray
  (245, 240, 225),   # beige / cream
  (250, 230, 235),   # soft pink
  (230, 245, 235),   # soft mint green
  (250, 250, 225)    # soft pale yellow
]
```

Background names used in filenames:

| RGB | Name |
|---|---|
| (255,255,255) | white |
| (235,235,235) | softgray |
| (245,240,225) | beige |
| (250,230,235) | softpink |
| (230,245,235) | mint |
| (250,250,225) | softyellow |

## Brightness & Saturation Augmentations

To model perceptual-level variation, each image uses:

- Saturation levels: ` [0.8, 1.0, 1.2] `
- Brightness levels: ` [0.8, 1.0, 1.2] `

These produce subtle but meaningful visual variations.

## Quality Control — Contrast Filtering

Images are only kept if:

`RGB_distance(ink_color, background_color) ≥ 40`

This guarantees:

- high readability
- no text–background color collision
- clean controlled stimuli

## Text Rendering Protocol

- Text rendered as UPPERCASE (e.g., `YELLOW`)
- Font: Arial 140pt
- Resolution: 512 × 512 px
- Centered using bounding-box metrics

This maintains a controlled and consistent structure across images.

## Dataset Size Breakdown

Total combinatorial space:

```
10 base colors
× 5 tone variants
× 6 backgrounds
× 3 saturation levels
× 3 brightness levels
= 2700 potential samples
```

After contrast filtering:

~2300–2600 usable images

## Filename Structure (Safe, Meaningful, No Dots)

Each image uses a fully descriptive filename, safe for all filesystems:

```
WORD_as_INK_bg-BGNAME_Txx_Sxx_Bxx.png
```

Encoding:

- Tone: 0.6 → `T06`, 0.8 → `T08`, 1.0 → `T10`, 1.2 → `T12`, 1.4 → `T14`
- Saturation: `S08`, `S10`, `S12`
- Brightness: `B08`, `B10`, `B12`

Examples:

```
YELLOW_as_red_bg-softpink_T08_S12_B10.png
GRAY_as_black_bg-mint_T14_S08_B12.png
GREEN_as_orange_bg-white_T06_S10_B08.png
```

This naming scheme is ideal for:

- model evaluation
- dataset parsing
- grouping by conflict type
- reproducibility

## Directory Structure

Recommended project layout:

```
project_root/
│
├── data/
│   └── stroop_dataset/
│       ├── YELLOW_as_red_bg-softpink_T08_S12_B10.png
│       ├── BLUE_as_green_bg-beige_T12_S10_B12.png
│       ├── ...
│       └── metadata.csv
│
└── src/
    └── generate_stroop_dataset.py
```

## Metadata File

A `metadata.csv` file is generated with the following columns:

| Column | Description |
|---|---|
| `filename` | image file name |
| `word` | displayed word (UPPERCASE) |
| `ink_rgb` | ink color after tone application |
| `background_rgb` | background color |
| `tone_factor` | tone multiplier |
| `saturation` | applied saturation level |
| `brightness` | applied brightness level |
