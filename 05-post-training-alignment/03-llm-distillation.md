# LLM distillation


**Distillation** trains a smaller **student** to imitate a larger **teacher** (or a stronger checkpoint) using the teacher’s **soft probabilities** or **hidden states**, not only hard labels. Goal is **cheaper inference** with acceptable quality—not the same objective as RLHF, though both are post-training tools.

1. **Teacher forward**  
   Run the big model on training prompts. Capture **logits z_T** over the vocabulary (possibly temperature-scaled).

2. **Student forward**  
   Run the small model, get logits **z_S**.

3. **Distillation loss**  
   Common term: **KL divergence** between softened teacher distribution and student distribution on each token (plus optional standard cross-entropy to ground truth tokens).

4. **Why it differs from RLHF**  
   - Distillation: “**match a teacher’s behavior distribution**” for compression or speed.  
   - RLHF: “**optimize human preference scores**” for alignment.  
   They can be combined in complex pipelines but solve different primary problems.

5. **Variants**  
   **Sequence-level** distillation (imitate rationales), **feature distillation** (match intermediate representations), **on-policy** distillation from teacher samples.


```mermaid
flowchart LR
  prompt[Prompt] --> teacher[Teacher_LLM_large]
  teacher --> soft[SoftTargets_logits]
  prompt --> student[Student_LLM_small]
  student --> logitsS[Student_logits]
  soft --> loss[DistillationLoss]
  logitsS --> loss
  loss --> update[UpdateStudentWeights]
```


```mermaid
flowchart TB
  distill[Distillation_focus_compress] --> goalA[LowerLatencyCost]
  rlhf[RLHF_focus_align] --> goalB[HumanPreferencesSafety]
```


## Extras

- **Capacity gap**: if the student is too small, it cannot fit the teacher distribution—curriculum and task narrowing help.
- **Data efficiency**: distillation can use **unlabeled** prompts if teacher provides targets; quality depends on teacher biases.
- **Legal/ethical**: cloning proprietary teachers without permission is a policy issue, not just a technical one.


## Terms

| Term | Meaning |
|------|---------|
| Soft targets | Probability vector over tokens from teacher softmax. |
| Student | Smaller model being trained to imitate teacher. |


That is the end of the sequence I wrote. Revisit the [repo index](../README.md) or skim [pre-training again](../01-introduction/01-pre-training-and-fine-tuning.md) if you want a second pass over the denser bullets.
