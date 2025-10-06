# Dataset Overview

This folder contains datasets used to evaluate robustness and failure modes in Large Language Models (LLMs).

## Structure

- `adversarial_prompts.csv` — Prompts designed to induce confusing or incorrect responses, including paradoxes, jailbreaks, misleading statements.
- `ambiguous_prompts.csv` — Prompts with multiple valid interpretations or vague instructions.
- `bias_prompts.csv` — Prompts testing for social, cultural, or ethical biases and stereotypes.
- `original_sources/` — Folder containing raw sources or inspiration (links, txt files).

## Guidelines

- Each `.csv` file contains two columns:
    - `prompt`  — The text prompt to send to the LLM
    - `category` — One of: adversarial, ambiguous, bias

## Dataset Sources

- [TruthfulQA Benchmark](https://github.com/sylinrl/TruthfulQA)
- [RealToxicityPrompts](https://github.com/allenai/real-toxicity-prompts)
- Custom prompts inspired by research literature

## Contribution

If new prompts are added:
- Follow CSV format as above
- Add source or rationale in a comment or supplementary file (optional)