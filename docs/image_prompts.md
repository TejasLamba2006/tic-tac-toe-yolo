# Visual Asset Prompts

Image generation prompts for documentation architecture diagrams.
These prompts are designed for AI image generation agents.

---

## Prompt 1: Hexagonal Architecture Diagram (Primary)

```image-prompt
A clean, professional technical architecture diagram in a hexagonal honeycomb layout on a white background.

Center hexagon: A glowing blue box labeled "ST Edge AI Pipeline" with a small neural network icon inside.

Surrounding hexagons (clockwise from top):
- "PyTorch/TensorFlow Model" with a flame icon (representing training)
- "STM32AI DC Cloud" with a cloud and microchip icon
- "NBG Optimizer" with a gear and binary code icon
- "STM32MP1 (Cortex-A7)" with a CPU chip icon
- "Yocto Linux" with a penguin and build icon
- "OpenSTLinux + X-LINUX-RBT1" with a package box icon
- "Deployment Target" with a rocket launch icon

Style: Flat vector illustration, corporate tech presentation style, dark blue and gold accent colors, clean lines, no gradients, white background, similar to STMicroelectronics official documentation aesthetic. Include subtle connection arrows between hexagons showing data flow. Add a small "Yocto Project Compatible" badge at the bottom.
```

## Prompt 2: Pipeline Flow Diagram (Secondary)

```image-prompt
A horizontal 4-step pipeline flow diagram on a light gray background.

Step 1: "Upload" — A hand placing a .tflite file into a cloud upload box. Color: blue.
Step 2: "Analyze" — A magnifying glass over a neural network graph. Color: teal.
Step 3: "Generate NBG" — A factory machine outputting a compact golden cube labeled "NBG". Color: gold.
Step 4: "Deploy" — A golden cube being placed onto an STM32MP1 board with Linux penguin. Color: green.

Style: Isometric 3D illustration, soft shadows, clean modern SaaS documentation style, each step connected by a dotted arrow. Include small status indicators (checkmarks) at each stage. STMicroelectronics brand colors (blue #001F5C, cyan #00B8E6, gold #FFC107).
```

## Prompt 3: MPU vs MCU Decision Tree (Tertiary)

```image-prompt
A decision tree / flowchart diagram on white background.

Start node: "Have a trained model?" → Yes
Branch 1: "Target is STM32MP1 (Cortex-A) with Linux?" → Yes → "Use NBG + STM32AI DC Cloud" → "Deploy via Yocto"
Branch 2: "Target is STM32F4/F7/H7 (Cortex-M)?" → Yes → "Use C-code generation + STM32Cube.AI"

Style: Simple flat vector flowchart, rounded rectangles, decision diamonds, clean arrows, minimal color palette (blue for process, gold for decision, green for end states). Include small icons: Linux penguin for MPU branch, microcontroller chip for MCU branch.
```
