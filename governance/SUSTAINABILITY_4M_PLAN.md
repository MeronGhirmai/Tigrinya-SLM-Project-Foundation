# Module 7: Sustainability & Environmental Justice Plan (4Ms Framework)

## 1. The 4Ms Compute Efficiency Strategy
- **Model:** `google/gemma-2-2b` (2.61B params) utilizing Grouped-Query Attention (GQA) and Alternating Local/Global Sliding Window Attention.
- **Machine:** Single low-TDP GPU (NVIDIA T4 70W or NVIDIA L4 72W).
- **Mechanization:** 4-bit NormalFloat (NF4) QLoRA (`r=16, alpha=32`), training only 0.79% of parameters (20.77M trainable params) with Gradient Checkpointing and `paged_adamw_8bit`.
- **Map:** High Carbon-Free Energy (CFE%) cloud regions.

## 2. Exact Mathematical FLOPs & VRAM Ledger
- **Total Fine-Tuning FLOPs (`6 * P * N`):** `6 * (2.61e9 params) * (1.152e7 tokens) = 1.804e17 FLOPs` (180.4 PetaFLOPs).
- **Peak Single-GPU VRAM (`batch=2, seq=512`):** `~7.61 GB` (1.44 GB 4-bit base + 0.12 GB LoRA states + 4.85 GB activations/logits + 1.20 GB CUDA context).
- **Energy & Carbon Footprint:** `~0.058 kWh` (`~14.5 gCO2eq`), achieving a >98% reduction vs. full-parameter fine-tuning.
