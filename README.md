# Evaluating Hallucinations in Text-to-Image Models for Procedural Knowledge Generation

This project investigates the ability of state-of-the-art text-to-image models to generate step-by-step visual content for procedural tasks in the recipe domain. The goal is to identify hallucinations and compare the capabilities of different generative models.

## Objectives

- Evaluate the performance of text-to-image models in generating coherent visual sequences
- Measure:
  - Step-image alignment
  - Visual consistency between steps
- Compare these metrics across multiple models



## Dataset

- **Domain**: Cooking recipes  
- **Source**: [AllRecipes](https://www.allrecipes.com/)  
- **Details**:
  - 250 recipe tasks
  - Each task contains 4–6 natural-language steps
  - Based on the refined dataset from the paper 'Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks', with additional modifications for model compatibility



## Text-to-Image Models

| Model Name             | Description |
|------------------------|-------------|
| **Stacked Diffusion**  | based on the paper 'Generating Illustrated Instructions with Stacked Diffusion Models' |
| **Stable Diffusion 2.1 (SD2.1)** | Open-source diffusion model used as a strong baseline |
| **Flux-1**             | Developed by **Black Forest Labs**, optimized for fast generation |



## Image Generation Pipeline

- For each of the 250 tasks, generate one image per step using each model.
- Non-stacked models generate images sequentially using a loop.
- Stacked Diffusion uses a hierarchical process to generate context-aware image sequences.



## Evaluation Metrics
### Automatic Evaluation

| Metric        | Purpose                                      | Tool/Library                                                  |
|---------------|----------------------------------------------|---------------------------------------------------------------|
| **CLIPScore** | Text-image semantic alignment                | [CLIP](https://openai.com/research/clip)         |
| **DreamSim**  | Visual consistency between sequential steps  | [DreamSim](https://dreamsim.mit.edu/)                         |
| **VQA Score** | Image's ability to answer step-related questions | [ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) |

### Human Annotation
Annotated 20 samples on the following criterias :
 
 **1. Described Action Visible**	
Is the main action described by the model actually visible or happening in the image?		
Yes: The action is clearly shown.		
Example: "Dice the carrots" — and a carrot is being diced.		
No: The action is not happening or cannot be seen.		
Example: The  step text says "Whisk the egg," but image shows whole eggs not cracked — mark No.		
		
  **2. Key Objects Present**		
Are the important objects or entities mentioned in the step text present and identifiable in the image?			
Yes: All or most important objects are there.		
Example: "Add chocolate chips into the boiling milk" — and both milk and chocoloate chips are present.		
No: Key objects are missing, wrong, or unclear.		
Example: Step text says "Pour milk into the pot," but the pot is empty.		
		
 **3. Hallucinated Content**		
Does the model mention anything that is not actually in the image (i.e., invented or imagined)?		
Yes: The model includes false or imagined details.		
Example: The step text says "Grease the tray with oil" , and the image shows a tray with muffins 

### Ground Truth Comparison 
The same  20 samples were comapred against ground truth images using Dreamsim Similarity score.
## Results
### Automatic Evaluation
| Model               | DreamSim Score(↑) | CLIP Score(↑) | VQA Score(↑) |
|--------------------|----------------|------------|-----------|
| **Stacked Diffusion** | **0.47**        | 0.27       | 0.21      |
| **SD2.1**             | 0.41           | **0.31**    | **0.23**  |
| **Flux-1**      | 0.44           | 0.30       | 0.22      |


- Stacked Diffusion excels at maintaining visual consistency across steps (DreamSim = 0.47), making it well-suited for producing coherent multi-step visuals.
- SD2.1 shows the strongest alignment with textual instructions, achieving the highest CLIPScore (0.31) and VQA Score (0.23), indicating it better grounds each image in the corresponding step description.
- Flux-1 offers a balanced performance, with moderate scores across all metrics, suggesting a trade-off between coherence and instruction alignment.


### Human Annotation
| Model               | Described Action Visible(↑) | Key Objects Present(↑) | Hallucinated Content(↓) 
|--------------------|----------------|------------|-----------|
| **Stacked Diffusion** | 0.82       | 0.83      | **0.11**       |
| **SD2.1**             | 0.71          | 0.82    | 0.27 |
| **Flux-1**      | **0.88**           | **0.87**       | 0.28     |

- Flux-1 demonstrates strong step-level fidelity, achieving the highest scores for both Described Action Visible (0.88) and Key Objects Present (0.87). Qualitatively, it generates high-clarity images that align closely with individual step descriptions. However, it lacks global consistency: in cases where the step text is ambiguous (e.g., "bake it in the oven for 30 minutes"), Flux-1 may introduce contextually incorrect objects (e.g., rendering a cake instead of the intended bread), reflecting limited integration of preceding step information.

- Stacked Diffusion achieves the lowest rate of hallucinated content (0.11), indicating strong textual grounding and high reliability in avoiding extraneous elements. Nevertheless, it exhibits weaker step-level alignment compared to Flux-1; for instance, actions such as "spread butter on the bread" may be rendered with less visual precision or incomplete object depiction, suggesting a trade-off between faithfulness and visual quality.

<img src="assets/sample_result.png" alt="Example output" width="50%">

### Ground Truth Comparison 
| Model               | DreamSim Score(↑)| 
|--------------------|----------------|
| **Stacked Diffusion** | 0.40      | 
| **SD2.1**             | 0.44          |
| **Flux-1**      | **0.46**           | 

## References

- [_Generating Coherent Sequences of Visual Illustrations for Real-World Manual Tasks_](https://openreview.net/forum?id=H1lFqT4YwS)  
- [_Generating Illustrated Instructions with Stacked Diffusion Models_](https://arxiv.org/abs/2306.16431)  
- [_CLIP: Connecting Text and Images_](https://openai.com/research/clip)  
- [_DreamSim: Consistency Metrics for Visual Sequences_](https://dreamsim.mit.edu/)  
