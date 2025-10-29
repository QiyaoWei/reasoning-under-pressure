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
DEFAULT_LOG_HYPERPARAMS="true"
DEFAULT_LOG_TO_WANDB="true"
DEFAULT_REBALANCE_MONITOR_REWARD="false"

# Parse command line arguments
CONFIG_FILE=""
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
LOG_HYPERPARAMS=""
LOG_TO_WANDB=""
REBALANCE_MONITOR_REWARD=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
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
        --log-hyperparams)
            LOG_HYPERPARAMS="$2"
            shift 2
            ;;
        --log-to-wandb)
            LOG_TO_WANDB="$2"
            shift 2
            ;;
        --rebalance-monitor-reward)
            REBALANCE_MONITOR_REWARD="$2"
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
    cat << EOF
Usage: $0 [OPTIONS]

Configuration:
  --config FILE               Load settings from JSON config file
                              Command-line args override config file values

Dataset Options:
  --dataset NAME              Dataset to use (default: $DEFAULT_DATASET)
                              Options: diamonds-seed0..7, function_correctness
  --experiment-name NAME      Experiment name (default: qwen2.5_coder_7b_\${dataset})

Training Options:
  --epochs NUM                Training epochs (default: $DEFAULT_TOTAL_EPOCHS)
  --lr RATE                   Learning rate (default: $DEFAULT_LR)
  --kl-coef COEFFICIENT       KL divergence coefficient (default: $DEFAULT_KL_COEF)
  --k COEFFICIENT             Verbosity reward coefficient (default: $DEFAULT_K)
  --resume-from-path PATH     Resume from checkpoint (default: none)

Monitor Options:
  --train-with-monitor BOOL   Enable monitor-based training (default: $DEFAULT_TRAIN_WITH_MONITOR)
  --monitor-correct-reward    Reward for correct monitor (default: $DEFAULT_MONITOR_CORRECT_REWARD)
  --monitor-wrong-reward      Reward for wrong monitor (default: $DEFAULT_MONITOR_WRONG_REWARD)
  --monitor-model-name        Monitor model name (default: $DEFAULT_MONITOR_MODEL_NAME)
  --rebalance-monitor-reward  Rebalance monitor reward (default: $DEFAULT_REBALANCE_MONITOR_REWARD)

Logging Options:
  --log-hyperparams BOOL      Save hyperparameters (default: $DEFAULT_LOG_HYPERPARAMS)
  --log-to-wandb BOOL         Log to wandb (default: $DEFAULT_LOG_TO_WANDB)

Other:
  --help, -h                  Show this help

Examples:
  $0                                                    # Use all defaults
  $0 --config my_config.json                            # Load from config file
  $0 --config my_config.json --lr 1e-5                  # Config + override
  $0 --dataset function_correctness                     # Different dataset
  $0 --k -0.001 --lr 1e-5 --kl-coef 0.2                 # Custom hyperparameters
  $0 --dataset diamonds-seed1 --epochs 2                # Quick test run
  $0 --resume-from-path checkpoints/model/global_step_60 # Resume training
  $0 --train-with-monitor true --monitor-correct-reward -2.0 # Monitor training
EOF
    exit 0
fi

# Load configuration from file if provided
if [ -n "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE..."

    # Check if file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required to parse JSON config files"
        echo "Install it with: brew install jq (macOS) or apt-get install jq (Linux)"
        exit 1
    fi

    # Parse config file and set variables (only if not already set via command line)
    [ -z "$DATASET_NAME" ] && DATASET_NAME=$(jq -r '.dataset // empty' "$CONFIG_FILE")
    [ -z "$EXPERIMENT_NAME" ] && EXPERIMENT_NAME=$(jq -r '.experiment_name // empty' "$CONFIG_FILE")
    [ -z "$EPOCHS" ] && EPOCHS=$(jq -r '.epochs // empty' "$CONFIG_FILE")
    [ -z "$K" ] && K=$(jq -r '.k // empty' "$CONFIG_FILE")
    [ -z "$LEARNING_RATE" ] && LEARNING_RATE=$(jq -r '.learning_rate // empty' "$CONFIG_FILE")
    [ -z "$KL_COEF" ] && KL_COEF=$(jq -r '.kl_coef // empty' "$CONFIG_FILE")
    [ -z "$RESUME_FROM_PATH" ] && RESUME_FROM_PATH=$(jq -r '.resume_from_path // empty' "$CONFIG_FILE")
    [ -z "$TRAIN_WITH_MONITOR" ] && TRAIN_WITH_MONITOR=$(jq -r '.train_with_monitor // empty' "$CONFIG_FILE")
    [ -z "$MONITOR_CORRECT_REWARD" ] && MONITOR_CORRECT_REWARD=$(jq -r '.monitor_correct_reward // empty' "$CONFIG_FILE")
    [ -z "$MONITOR_WRONG_REWARD" ] && MONITOR_WRONG_REWARD=$(jq -r '.monitor_wrong_reward // empty' "$CONFIG_FILE")
    [ -z "$MONITOR_MODEL_NAME" ] && MONITOR_MODEL_NAME=$(jq -r '.monitor_model_name // empty' "$CONFIG_FILE")
    [ -z "$LOG_HYPERPARAMS" ] && LOG_HYPERPARAMS=$(jq -r '.log_hyperparams // empty' "$CONFIG_FILE")
    [ -z "$LOG_TO_WANDB" ] && LOG_TO_WANDB=$(jq -r '.log_to_wandb // empty' "$CONFIG_FILE")
    [ -z "$REBALANCE_MONITOR_REWARD" ] && REBALANCE_MONITOR_REWARD=$(jq -r '.rebalance_monitor_reward // empty' "$CONFIG_FILE")

    echo "Configuration loaded successfully"
fi

# Set defaults if not provided via command line or config file
DATASET_NAME=${DATASET_NAME:-$DEFAULT_DATASET}
K=${K:-$DEFAULT_K}
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}
KL_COEF=${KL_COEF:-$DEFAULT_KL_COEF}
EPOCHS=${EPOCHS:-$DEFAULT_TOTAL_EPOCHS}
TRAIN_WITH_MONITOR=${TRAIN_WITH_MONITOR:-$DEFAULT_TRAIN_WITH_MONITOR}
MONITOR_CORRECT_REWARD=${MONITOR_CORRECT_REWARD:-$DEFAULT_MONITOR_CORRECT_REWARD}
MONITOR_WRONG_REWARD=${MONITOR_WRONG_REWARD:-$DEFAULT_MONITOR_WRONG_REWARD}
MONITOR_MODEL_NAME=${MONITOR_MODEL_NAME:-$DEFAULT_MONITOR_MODEL_NAME}
LOG_HYPERPARAMS=${LOG_HYPERPARAMS:-$DEFAULT_LOG_HYPERPARAMS}
LOG_TO_WANDB=${LOG_TO_WANDB:-$DEFAULT_LOG_TO_WANDB}
REBALANCE_MONITOR_REWARD=${REBALANCE_MONITOR_REWARD:-$DEFAULT_REBALANCE_MONITOR_REWARD}
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
echo "Log hyperparameters: $LOG_HYPERPARAMS"
echo "Log to wandb: $LOG_TO_WANDB"
echo "Rebalance monitor reward: $REBALANCE_MONITOR_REWARD"

# Determine if KL loss should be used based on coefficient
case "$KL_COEF" in
    0|0.0|0.00)
        USE_KL_LOSS="False"
        echo "KL coefficient is 0, disabling KL loss"
        ;;
    *)
        USE_KL_LOSS="True"
        echo "KL coefficient is $KL_COEF, enabling KL loss"
        ;;
esac


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
export REBALANCE_MONITOR_REWARD

# Log hyperparameters before training if enabled
if [ "$LOG_HYPERPARAMS" = "true" ]; then
    echo "Logging hyperparameters..."
    
    WANDB_FLAG=""
    if [ "$LOG_TO_WANDB" = "true" ]; then
        WANDB_FLAG="--log-to-wandb"
    fi
    
    python3 log_hyperparameters.py \
        --experiment-name "${EXPERIMENT_NAME}" \
        --dataset-name "${DATASET_NAME}" \
        --train-file "${TRAIN_FILE}" \
        --val-file "${VAL_FILE}" \
        --max-response-length "${MAX_RESPONSE_LENGTH}" \
        --learning-rate "${LEARNING_RATE}" \
        --epochs "${EPOCHS}" \
        --kl-coef "${KL_COEF}" \
        --use-kl-loss "${USE_KL_LOSS}" \
        --resume-mode "${RESUME_MODE}" \
        --resume-from-path "${RESUME_FROM_PATH}" \
        ${WANDB_FLAG} \
        --wandb-project "verl_grpo_${DATASET_NAME}"
    
    if [ $? -ne 0 ]; then
        echo "Warning: Hyperparameter logging failed, but continuing with training..."
    fi
fi

echo "Starting GRPO training..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=rewards.py \
    custom_reward_function.name=compute_score \
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
    actor_rollout_ref.ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.validation_data_dir=./checkpoints/verl_grpo_${DATASET_NAME}/${EXPERIMENT_NAME}/validation_outputs \
    trainer.project_name="verl_grpo_${DATASET_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path=${RESUME_FROM_PATH} \
    $@