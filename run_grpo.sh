#!/bin/bash
# GRPO training script for diamonds and function correctness datasets using VERL
# Based on the GSM8K example but adapted for diamonds and function correctness tasks

set -x

# Default values
DEFAULT_DATASET="diamonds-seed0"
DEFAULT_TOTAL_EPOCHS=10
DEFAULT_K=0.0
DEFAULT_LR=5e-6
DEFAULT_KL_COEF=0.0
DEFAULT_TRAIN_WITH_MONITOR="false"
DEFAULT_MONITOR_CORRECT_REWARD="-1.0"
DEFAULT_MONITOR_WRONG_REWARD="1.0"
DEFAULT_MONITOR_MODEL_NAME="gpt-4o-mini"

# Parse command line arguments
DATASET_NAME=""
EXPERIMENT_NAME=""
EPOCHS=""
K=""
LEARNING_RATE=""
KL_COEF=""
RESUME_FROM_PATH=""
TRAIN_WITH_MONITOR=""
MONITOR_CORRECT_REWARD=""
MONITOR_WRONG_REWARD=""
MONITOR_MODEL_NAME=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --kl-coef)
            KL_COEF="$2"
            shift 2
            ;;
        --resume-from-path)
            RESUME_FROM_PATH="$2"
            shift 2
            ;;
        --train-with-monitor)
            TRAIN_WITH_MONITOR="$2"
            shift 2
            ;;
        --monitor-correct-reward)
            MONITOR_CORRECT_REWARD="$2"
            shift 2
            ;;
        --monitor-wrong-reward)
            MONITOR_WRONG_REWARD="$2"
            shift 2
            ;;
        --monitor-model-name)
            MONITOR_MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset NAME              Dataset name (default: $DEFAULT_DATASET)"
    echo "                              Supported: diamonds-seed0 to diamonds-seed7, function_correctness"
    echo "  --k COEFFICIENT             Verbosity reward coefficient (default: $DEFAULT_K)"
    echo "                              Negative values penalize verbose responses"
    echo "  --experiment-name NAME      Custom experiment name (default: qwen2.5_coder_7b_\${dataset})"
    echo "  --epochs NUM                Number of epochs (default: $DEFAULT_TOTAL_EPOCHS)"
    echo "  --lr RATE                   Learning rate (default: $DEFAULT_LR)"
    echo "  --kl-coef COEFFICIENT       KL divergence coefficient (default: $DEFAULT_KL_COEF)"
    echo "  --resume-from-path PATH     Resume training from checkpoint path (default: null)"
    echo "  --train-with-monitor BOOL   Enable monitor training (default: $DEFAULT_TRAIN_WITH_MONITOR)"
    echo "  --monitor-correct-reward    Reward when monitor is correct (default: $DEFAULT_MONITOR_CORRECT_REWARD)"
    echo "  --monitor-wrong-reward      Reward when monitor is wrong (default: $DEFAULT_MONITOR_WRONG_REWARD)"
    echo "  --monitor-model-name        OpenAI model for monitor (default: $DEFAULT_MONITOR_MODEL_NAME)"
    echo "  --help, -h                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                         # Use defaults"
    echo "  $0 --dataset function_correctness          # Use function_correctness dataset"
    echo "  $0 --k -0.001                              # Set verbosity reward"
    echo "  $0 --lr 1e-5 --kl-coef 0.2                 # Set learning rate and KL coefficient"
    echo "  $0 --experiment-name my_experiment          # Custom experiment name"
    echo "  $0 --dataset diamonds-seed1 --k 0.005 --lr 2e-6 --epochs 2  # Multiple options"
    echo "  $0 --resume-from-path checkpoints/model/global_step_60  # Resume from checkpoint"
    echo "  $0 --train-with-monitor true --monitor-correct-reward -2.0  # Enable monitor training"
    echo "  $0 --monitor-model-name o3  # Use different monitor model"
    exit 0
fi

# Set defaults if not provided via command line
DATASET_NAME=${DATASET_NAME:-$DEFAULT_DATASET}
K=${K:-$DEFAULT_K}
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}
KL_COEF=${KL_COEF:-$DEFAULT_KL_COEF}
EPOCHS=${EPOCHS:-$DEFAULT_TOTAL_EPOCHS}
TRAIN_WITH_MONITOR=${TRAIN_WITH_MONITOR:-$DEFAULT_TRAIN_WITH_MONITOR}
MONITOR_CORRECT_REWARD=${MONITOR_CORRECT_REWARD:-$DEFAULT_MONITOR_CORRECT_REWARD}
MONITOR_WRONG_REWARD=${MONITOR_WRONG_REWARD:-$DEFAULT_MONITOR_WRONG_REWARD}
MONITOR_MODEL_NAME=${MONITOR_MODEL_NAME:-$DEFAULT_MONITOR_MODEL_NAME}

# Set max_response_length based on dataset
if [[ "$DATASET_NAME" == function_correctness ]]; then
    MAX_RESPONSE_LENGTH=2048
else
    MAX_RESPONSE_LENGTH=1024
fi

# Set experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="qwen2.5_coder_7b_${DATASET_NAME}"
fi

# Set data paths based on dataset name
DATA_DIR=./data/${DATASET_NAME}
TRAIN_FILE=${DATA_DIR}/train.parquet
VAL_FILE=${DATA_DIR}/val.parquet

# Verify files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    echo "Please ensure the dataset has been prepared using prepare_diamonds_dataset.py"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation file not found: $VAL_FILE"
    echo "Please ensure the dataset has been prepared using prepare_diamonds_dataset.py"
    exit 1
fi

echo "Experiment name: $EXPERIMENT_NAME"
echo "Using dataset: $DATASET_NAME"
echo "Training file: $TRAIN_FILE"
echo "Validation file: $VAL_FILE"
echo "Number of epochs: $EPOCHS"
echo "Resume from path: $RESUME_FROM_PATH"
echo "Learning rate: $LEARNING_RATE"
echo "Max response length: $MAX_RESPONSE_LENGTH"
echo "KL coefficient: $KL_COEF"
echo "Verbosity reward coefficient K: $K"
echo "Train with monitor: $TRAIN_WITH_MONITOR"
if [ "$TRAIN_WITH_MONITOR" = "true" ]; then
    echo "Monitor model name: $MONITOR_MODEL_NAME"
    echo "Monitor correct reward: $MONITOR_CORRECT_REWARD"
    echo "Monitor wrong reward: $MONITOR_WRONG_REWARD"
fi


# Determine if KL loss should be used based on coefficient
USE_KL_LOSS="True"
if (( $(echo "$KL_COEF == 0.0" | bc -l) )); then
    USE_KL_LOSS="False"
    echo "KL coefficient is 0.0, disabling KL loss"
fi

# Set resume mode depending on whether resume from path is provided
if [ -z "$RESUME_FROM_PATH" ]; then
    RESUME_MODE=disable
else
    RESUME_MODE=resume_path
fi

# Export environment variables for the reward function
export K
export TRAIN_WITH_MONITOR
export MONITOR_CORRECT_REWARD
export MONITOR_WRONG_REWARD
export MONITOR_MODEL_NAME

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="verl_grpo_${DATASET_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path=${RESUME_FROM_PATH} $@