# 02 — Quantization overview

## In one minute

**Quantization** stores weights (and sometimes activations) in **fewer bits** than FP32/FP16 so the model uses less memory, runs faster on many devices, and costs less to host—at the price of careful scaling so quality stays acceptable.

## Beginner walkthrough

1. **What is being reduced?**  
   Neural nets are arrays of numbers. By default training often uses **FP32** (32-bit float) or **FP16/BF16** (16-bit). Quantization maps many of those values to **int8** or even **int4** for storage and some math paths.

2. **Why it matters for LLMs**  
   A 7B or 70B model has billions of weights. Cutting average storage from 16 bits to 4–8 bits can shrink **disk and RAM** a lot, which also improves **batch size** and **latency** on GPUs and enables **edge** devices.

3. **What you trade**  
   Fewer bits means **rounding error**. Aggressive quantization without calibration or QAT can hurt accuracy; later folders show how engineers recover quality.

4. **Weights vs activations**  
   **Weight quantization** is most common in LLM compression. **Activation quantization** saves memory during inference but is trickier because activations change per input.

## Visuals

**High-level compression path**

```mermaid
flowchart LR
  hiPrec[HighPrecisionWeights] --> quantMap[QuantizeMapping]
  quantMap --> loPrec[LowBitStorage]
```

**Dtypes at a glance (typical roles)**

| Storage / compute | Bits | Typical role |
|-------------------|------|----------------|
| FP32 | 32 | Reference training numerics; heavy. |
| FP16 / BF16 | 16 | Common training and inference on GPU. |
| INT8 | 8 | Inference speedups; PTQ/QAT targets. |
| INT4 | 4 | Aggressive LLM weight packing (often with methods like NF4 in practice). |

## Going deeper

- **Per-tensor vs per-channel**: one scale per whole tensor is simple; per-channel (per output channel) often preserves accuracy for conv layers; LLM linear layers use schemes like **per-group** scales in GPTQ/AWQ-style methods.
- **Symmetric vs asymmetric**: next two folders—symmetric is simpler; asymmetric handles biased ranges (e.g. mostly positive with a small negative tail).
- **Fake quant in QAT**: during training, values are rounded as if quantized but gradients still flow—folder 06.

## Mini glossary

| Term | Meaning |
|------|---------|
| FP32 / FP16 / BF16 | Floating-point formats with different exponent/mantissa tradeoffs. |
| INT8 / INT4 | Integer storage; values live in a mapped range with a scale (and maybe zero-point). |
| Calibration | Running representative data to pick ranges/scales before PTQ. |

## What to read next

**[03 — Symmetric quantization](02-symmetric-quantization.md)** — the first concrete mapping from real numbers to integer bins.
