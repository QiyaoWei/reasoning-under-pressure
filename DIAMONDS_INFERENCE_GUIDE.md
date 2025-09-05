# DIAMONDS Dataset Inference Guide

This guide covers all the commands and workflows for running inference on the DIAMONDS dataset with VERL-trained models.

## 1. Running Inference

### Basic Inference Commands

```bash
# Run inference on a local checkpoint
python run_inference_diamonds.py \
    --checkpoint checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor \
    --num-samples 100 \
    --temperature 0.1 \
    --output-dir ./outputs

# Evaluate multiple checkpoints in a directory
python run_inference_diamonds.py \
    --checkpoints-dir checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds \
    --pattern "global_step_*" \
    --num-samples 100 \
    --output-dir ./outputs

# Run with more samples for final evaluation
python run_inference_diamonds.py \
    --checkpoint checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor \
    --num-samples 500 \
    --temperature 0.1
```

### Key Parameters
- `--num-samples`: Number of test samples to evaluate (default: 100)
- `--temperature`: Generation temperature (default: 0.1, use 0.0 for deterministic)
- `--output-dir`: Where to save predictions and results (default: ./outputs)
- `--device`: cuda or cpu (auto-detected by default)
- `--save-predictions`: Save individual predictions (default: True)

## 2. Converting VERL Checkpoints to HuggingFace Format

### During Training (Automatic Conversion)

Add this to your `run_diamonds_grpo.sh` training script to automatically save HuggingFace format:

```bash
# Add to training command
actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra', 'hf_model']"

# Or save ONLY HuggingFace format (saves space)
actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']"
```

### Post-Training Conversion (Manual)

Convert existing FSDP checkpoints to HuggingFace format:

```bash
# Convert a single checkpoint
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor \
    --target_dir checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor/huggingface

# Convert all checkpoints (example script)
for checkpoint in checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_*/; do
    echo "Converting $checkpoint"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$checkpoint/actor" \
        --target_dir "$checkpoint/actor/huggingface"
done
```

## 3. Using HuggingFace Hub Models

### Loading from HuggingFace Hub

```bash
# If model is at repository root
python run_inference_diamonds.py --checkpoint namespace/repo_name

# Example
python run_inference_diamonds.py --checkpoint BenatCambridge/verl
```

**Note**: HuggingFace expects models at the repository root, not in subdirectories.

### Downloading Subdirectories from HuggingFace

Use the provided download script:

```bash
# Download a specific checkpoint subdirectory
python download_hf_subdirectory.py \
    --repo BenatCambridge/verl \
    --subdirectory verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor/huggingface \
    --local-dir ./downloaded_checkpoint

# Then run inference
python run_inference_diamonds.py \
    --checkpoint ./downloaded_checkpoint/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor/huggingface
```

Or use huggingface-cli directly:

```bash
# Download specific directory with huggingface-cli
huggingface-cli download BenatCambridge/verl \
    --include "verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor/huggingface/*" \
    --local-dir ./downloaded_model \
    --local-dir-use-symlinks False
```

## 4. Training Script Reference

### Basic GRPO Training

```bash
# Run GRPO training on DIAMONDS dataset
./run_diamonds_grpo.sh

# Key parameters in the script:
# - data.train_files: Path to training data
# - data.val_files: Path to validation data  
# - actor_rollout_ref.model.path: Base model (e.g., Qwen/Qwen2.5-Coder-7B-Instruct)
# - trainer.n_gpus_per_node: Number of GPUs
# - trainer.total_epochs: Training epochs
# - trainer.save_freq: Checkpoint save frequency
```

## 5. Output Files

After running inference, you'll find:

```
outputs/
├── predictions/
│   ├── global_step_100_predictions.json  # Metrics and predictions
│   └── global_step_100_raw_outputs.json  # Raw model outputs
├── checkpoint_comparison.json  # If evaluating multiple checkpoints
└── checkpoint_comparison.csv   # CSV format for analysis
```

## 6. Common Workflows

### Full Evaluation Pipeline

```bash
# 1. Train model with HuggingFace export
./run_diamonds_grpo.sh  # Make sure hf_model is in save_contents

# 2. Evaluate best checkpoint
python run_inference_diamonds.py \
    --checkpoint checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor \
    --num-samples 500

# 3. Compare all checkpoints
python run_inference_diamonds.py \
    --checkpoints-dir checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds \
    --num-samples 100
```

### Quick Test Run

```bash
# Test with just 10 samples to verify everything works
python run_inference_diamonds.py \
    --checkpoint checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor \
    --num-samples 10 \
    --temperature 0.1
```

## 7. Troubleshooting

### Common Issues

1. **"Checkpoint not found" error with HuggingFace models**
   - Ensure the model is at the repository root on HuggingFace
   - Or download locally first using the download script

2. **Missing model weights in huggingface folder**
   - Run the model merger to convert FSDP checkpoints
   - Or add 'hf_model' to save_contents during training

3. **Out of memory during inference**
   - Reduce batch size (process one sample at a time)
   - Use CPU inference with --device cpu
   - Use a smaller test set with --num-samples

## 8. File Locations

- Training script: `/root/verl/run_diamonds_grpo.sh`
- Inference script: `/root/verl/run_inference_diamonds.py`
- Download helper: `/root/verl/download_hf_subdirectory.py`
- Checkpoints: `/root/verl/checkpoints/verl_grpo_diamonds/`
- Outputs: `/root/verl/outputs/` or specified with --output-dir