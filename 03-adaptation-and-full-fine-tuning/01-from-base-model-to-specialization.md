# 07 — From base model to specialization

## In one minute

A **base** model from large-scale pre-training becomes useful through **supervised** or **preference** tuning (chat assistants), then branches into **domain** specialization (finance, medicine) or **task** specialization (summarization, extraction, classification-style heads).

## Beginner walkthrough

1. **Base model**  
   Trained on broad data (Wikipedia-scale mixes, books, web, code). Strong **general** competence.

2. **Instruction / chat alignment (example layer)**  
   Often implemented with **supervised fine-tuning** on high-quality prompt–response pairs, sometimes followed by **RLHF** (folder 14). Produces models closer to **ChatGPT-style** assistants.

3. **Domain-specific fine-tuning**  
   Same architecture, data focused on one vertical: **healthcare**, **finance**, **automotive** manuals, etc.

4. **Task-specific fine-tuning**  
   Narrow objectives labeled A, B, C, D in your notes: e.g. JSON-only output, SQL generation, classification with a head, retrieval reranking.

5. **Why draw the diagram**  
   It clarifies **reuse**: one expensive base supports many downstream products if specialization is cheap enough (PEFT, quantization).

## Visuals

```mermaid
flowchart TB
  base[BaseModel_BroadPretrain]
  base --> sft[InstructionOrChatTuning]
  sft --> domain[DomainSpecific_FT]
  sft --> tasks[TaskSpecific_FT]
  domain --> d1[Finance]
  domain --> d2[Healthcare]
  domain --> d3[Automotive]
  tasks --> t1[TaskA]
  tasks --> t2[TaskB]
  tasks --> t3[TaskC]
  tasks --> t4[TaskD]
```

## Going deeper

- **Multi-stage** pipelines blur lines: you might CPT on domain text, then SFT, then RLHF.
- **Evaluation** must track both **target** metric and **regressions** on general skills (ties to catastrophic forgetting, folder 15).
- **Open vs closed weights**: specialization techniques apply to both; deployment constraints differ.

## Mini glossary

| Term | Meaning |
|------|---------|
| SFT | Supervised fine-tuning on input–output pairs. |
| Domain FT | Fine-tune predominantly on in-domain corpus. |

## What to read next

**[08 — Full-parameter fine-tuning](02-full-parameter-fine-tuning.md)** — what “train everything” really costs.
