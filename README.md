# Evaluating Hallucinations in Text-to-Image Models for Procedural Knowledge Generation

<img src="path/to/your/image.png" alt="Example output" width="100%">

This project investigates the ability of state-of-the-art text-to-image models to generate step-by-step visual content for procedural tasks in the recipe domain. It focuses on evaluating:

- Semantic alignment between text instructions and generated images
- Visual consistency across sequential steps

The goal is to identify hallucinations and compare the capabilities of different generative models.

---

## Objectives

- Evaluate the performance of text-to-image models in generating coherent visual sequences
- Measure:
  - Step-image alignment
  - Visual consistency between steps
- Compare these metrics across multiple models

---

## Dataset

- **Domain**: Cooking recipes  
- **Source**: [AllRecipes](https://www.allrecipes.com/)  
- **Details**:
  - 250 recipe tasks
  - Each task contains 4–6 natural-language steps
  - Based on the refined dataset from [_Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks_](https://openreview.net/forum?id=H1lFqT4YwS), with additional modifications for model compatibility

---

## Text-to-Image Models

| Model Name             | Description |
|------------------------|-------------|
| **Stacked Diffusion**  | From [*Generating Illustrated Instructions with Stacked Diffusion Models*](https://arxiv.org/abs/2306.16431) |
| **Stable Diffusion 2.1 (SD2.1)** | Open-source diffusion model used as a strong baseline |
| **Flux 1**             | Developed by **Black Forest Labs**, optimized for fast generation |

---

## Image Generation Pipeline

- For each of the 250 tasks, generate one image per step using each model.
- Non-stacked models generate images sequentially using a loop.
- Stacked Diffusion uses a hierarchical process to generate context-aware image sequences.

---

## Evaluation Metrics

| Metric       | Purpose                          | Tool/Library |
|--------------|----------------------------------|--------------|
| **CLIPScore** | Text-image semantic alignment     | [CLIP](https://openai.com/research/clip) / OpenCLIP |
| **DreamSim**  | Visual consistency between steps | [DreamSim](https://dreamsim.mit.edu/) |

---

## Results

| Model             | DreamSim Score | CLIPScore |
|------------------|----------------|-----------|
| **Stacked Diffusion** | 0.47           | 0.27      |
| **SD2.1**             | 0.41           | 0.31      |
| **Flux-1**            | 0.44           | 0.30      |

- Stacked Diffusion performs best in maintaining visual consistency.
- SD2.1 achieves the highest alignment with textual instructions.

---

## References

- [_Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks_](https://openreview.net/forum?id=H1lFqT4YwS)  
- [_Generating Illustrated Instructions with Stacked Diffusion Models_](https://arxiv.org/abs/2306.16431)  
- [_CLIP: Connecting Text and Images_](https://openai.com/research/clip)  
- [_DreamSim: Consistency Metrics for Visual Sequences_](https://dreamsim.mit.edu/)  
