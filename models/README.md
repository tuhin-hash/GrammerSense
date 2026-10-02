# Models & Checkpoints

This directory houses the neural network weights, pretrained model configuration, and fine-tuned checkpoints.

## Supported Operation Modes

### 1. Pretrained Mode (Default)
Immediately usable out-of-the-box using standard Sequence-to-Sequence models optimized for English Grammar Error Correction:
- **Default Checkpoint**: `vennify/t5-base-grammar-correction` (T5-base fine-tuned on C4 200M grammar error dataset).
- **Lightweight Alternative**: `t5-small` (for ultra-low memory or embedded environments).

### 2. Fine-Tuning Mode
Optional end-to-end training pipeline using `scripts/train.py`:
- Trains on deduplicated pairs in `data/processed/train.jsonl` with validation tracking on `data/processed/val.jsonl`.
- Saves the best checkpoint to `models/checkpoints/best_model/` along with tokenizer and `training_summary.json`.

## CPU Optimizations
- **Thread Tuning**: Configured with `torch.set_num_threads` (capped to 4) to eliminate core contention on multi-core CPUs.
- **Inference Mode**: Inference wrapped in `torch.no_grad()` to prevent memory overhead and activation tracking.
- **Length Capping**: Input and target lengths are trimmed to 128 tokens, ensuring fast sub-second to 1-2 second latency on CPU.
