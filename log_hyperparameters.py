#!/usr/bin/env python3
"""
Hyperparameter logging script for GRPO training.
Logs all hyperparameters used in training to both local YAML file and wandb.
"""

import os
import sys
import yaml
import argparse
import wandb
from datetime import datetime
from pathlib import Path


def create_hyperparameter_dict(args):
    """Create a comprehensive dictionary of all hyperparameters."""
    
    # Get environment variables that might be set by the bash script
    env_vars = {
        'K': os.environ.get('K', ''),
        'TRAIN_WITH_MONITOR': os.environ.get('TRAIN_WITH_MONITOR', ''),
        'MONITOR_CORRECT_REWARD': os.environ.get('MONITOR_CORRECT_REWARD', ''),
        'MONITOR_WRONG_REWARD': os.environ.get('MONITOR_WRONG_REWARD', ''),
        'MONITOR_MODEL_NAME': os.environ.get('MONITOR_MODEL_NAME', ''),
        'REBALANCE_MONITOR_REWARD': os.environ.get('REBALANCE_MONITOR_REWARD', ''),
    }
    
    # Convert string values to appropriate types
    def convert_value(value):
        if value == '':
            return None
        # Try to convert to float first, then int, then keep as string
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            return value
    
    # Clean up environment variables
    for key, value in env_vars.items():
        env_vars[key] = convert_value(value)
    
    # Create hyperparameter dictionary
    hyperparams = {
        'experiment_info': {
            'experiment_name': args.experiment_name,
            'dataset_name': args.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'script_version': '1.0',
        },
        
        'data_config': {
            'train_file': args.train_file,
            'val_file': args.val_file,
            'train_batch_size': 1024,
            'max_prompt_length': 2048,
            'max_response_length': args.max_response_length,
            'filter_overlong_prompts': True,
            'truncation': 'error',
        },
        
        'model_config': {
            'model_path': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
            'use_remove_padding': True,
            'enable_gradient_checkpointing': True,
        },
        
        'training_config': {
            'algorithm': 'grpo',
            'learning_rate': args.learning_rate,
            'total_epochs': args.epochs,
            'ppo_mini_batch_size': 256,
            'ppo_micro_batch_size_per_gpu': 32,
            'use_kl_loss': args.use_kl_loss,
            'kl_loss_coef': args.kl_coef,
            'kl_loss_type': 'low_var_kl',
            'entropy_coeff': 0,
            'critic_warmup': 0,
        },
        
        'rollout_config': {
            'log_prob_micro_batch_size_per_gpu': 32,
            'tensor_model_parallel_size': 2,
            'name': 'vllm',
            'gpu_memory_utilization': 0.6,
            'n': 4,
            'temperature': 1.0,
        },
        
        'reference_config': {
            'log_prob_micro_batch_size_per_gpu': 32,
            'fsdp_param_offload': True,
        },
        
        'fsdp_config': {
            'param_offload': False,
            'optimizer_offload': False,
        },
        
        'algorithm_config': {
            'use_kl_in_reward': False,
        },
        
        'trainer_config': {
            'n_gpus_per_node': 4,
            'nnodes': 1,
            'save_freq': 20,
            'test_freq': 5,
            'resume_mode': args.resume_mode,
            'resume_from_path': args.resume_from_path,
        },
        
        'logging_config': {
            'logger': ['console', 'wandb'],
            'project_name': f'verl_grpo_{args.dataset_name}',
            'experiment_name': args.experiment_name,
        },
        
        'reward_config': {
            'verbosity_coefficient_k': env_vars['K'],
            'train_with_monitor': env_vars['TRAIN_WITH_MONITOR'],
            'monitor_correct_reward': env_vars['MONITOR_CORRECT_REWARD'],
            'monitor_wrong_reward': env_vars['MONITOR_WRONG_REWARD'],
            'monitor_model_name': env_vars['MONITOR_MODEL_NAME'],
            'rebalance_monitor_reward': env_vars['REBALANCE_MONITOR_REWARD'],
        },
        
        'command_line_args': {
            'dataset': args.dataset_name,
            'epochs': args.epochs,
            'k': env_vars['K'],
            'experiment_name': args.experiment_name,
            'lr': args.learning_rate,
            'kl_coef': args.kl_coef,
            'resume_from_path': args.resume_from_path,
            'train_with_monitor': env_vars['TRAIN_WITH_MONITOR'],
            'monitor_correct_reward': env_vars['MONITOR_CORRECT_REWARD'],
            'monitor_wrong_reward': env_vars['MONITOR_WRONG_REWARD'],
            'monitor_model_name': env_vars['MONITOR_MODEL_NAME'],
            'rebalance_monitor_reward': env_vars['REBALANCE_MONITOR_REWARD'],
        }
    }
    
    return hyperparams


def save_hyperparameters_locally(hyperparams, output_dir):
    """Save hyperparameters to local YAML file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed YAML file
    yaml_file = os.path.join(output_dir, 'hyperparameters.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=False)
    
    # Save a simplified version for easy reading
    simple_file = os.path.join(output_dir, 'hyperparameters_simple.txt')
    with open(simple_file, 'w') as f:
        f.write("GRPO Training Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        
        # Key hyperparameters
        f.write("Key Hyperparameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset: {hyperparams['experiment_info']['dataset_name']}\n")
        f.write(f"Experiment: {hyperparams['experiment_info']['experiment_name']}\n")
        f.write(f"Learning Rate: {hyperparams['training_config']['learning_rate']}\n")
        f.write(f"Epochs: {hyperparams['training_config']['total_epochs']}\n")
        f.write(f"KL Coefficient: {hyperparams['training_config']['kl_loss_coef']}\n")
        f.write(f"Verbosity K: {hyperparams['reward_config']['verbosity_coefficient_k']}\n")
        f.write(f"Max Response Length: {hyperparams['data_config']['max_response_length']}\n")
        f.write(f"Use KL Loss: {hyperparams['training_config']['use_kl_loss']}\n")
        f.write(f"Train with Monitor: {hyperparams['reward_config']['train_with_monitor']}\n")
        if hyperparams['reward_config']['train_with_monitor']:
            f.write(f"Monitor Model: {hyperparams['reward_config']['monitor_model_name']}\n")
            f.write(f"Monitor Correct Reward: {hyperparams['reward_config']['monitor_correct_reward']}\n")
            f.write(f"Monitor Wrong Reward: {hyperparams['reward_config']['monitor_wrong_reward']}\n")
            f.write(f"Rebalance Monitor Reward: {hyperparams['reward_config']['rebalance_monitor_reward']}\n")
        f.write(f"Resume From: {hyperparams['trainer_config']['resume_from_path'] or 'None'}\n")
        f.write(f"Timestamp: {hyperparams['experiment_info']['timestamp']}\n")
    
    print(f"Hyperparameters saved to:")
    print(f"  - {yaml_file}")
    print(f"  - {simple_file}")
    
    return yaml_file, simple_file


# def log_to_wandb(hyperparams, project_name, experiment_name):
#     """Log hyperparameters to wandb."""
#     try:
#         # Initialize wandb run
#         wandb.init(
#             project=project_name,
#             name=experiment_name,
#             config=hyperparams,
#             reinit=True
#         )
        
#         # Log hyperparameters as a table for better visualization
#         config_table = wandb.Table(columns=["Category", "Parameter", "Value"])
        
#         for category, params in hyperparams.items():
#             if isinstance(params, dict):
#                 for param, value in params.items():
#                     config_table.add_data(category, param, str(value))
#             else:
#                 config_table.add_data("General", category, str(params))
        
#         wandb.log({"hyperparameters_table": config_table})
        
#         print(f"Hyperparameters logged to wandb project: {project_name}")
#         print(f"Experiment name: {experiment_name}")
        
#     except Exception as e:
#         print(f"Warning: Failed to log to wandb: {e}")
#         print("Continuing with local logging only...")


def main():
    parser = argparse.ArgumentParser(description="Log GRPO training hyperparameters")
    
    # Required arguments
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--dataset-name", required=True, help="Dataset name")
    parser.add_argument("--train-file", required=True, help="Training file path")
    parser.add_argument("--val-file", required=True, help="Validation file path")
    parser.add_argument("--max-response-length", type=int, required=True, help="Max response length")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--kl-coef", type=float, required=True, help="KL coefficient")
    parser.add_argument("--use-kl-loss", type=str, required=True, help="Use KL loss (True/False)")
    parser.add_argument("--resume-mode", required=True, help="Resume mode")
    parser.add_argument("--resume-from-path", help="Resume from path")
    
    # Optional arguments
    parser.add_argument("--output-dir", help="Output directory for hyperparameter files")
    parser.add_argument("--log-to-wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", help="Wandb project name")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = f"checkpoints/verl_grpo_{args.dataset_name}/{args.experiment_name}"
    
    # Create hyperparameter dictionary
    hyperparams = create_hyperparameter_dict(args)
    
    # Save locally
    yaml_file, simple_file = save_hyperparameters_locally(hyperparams, args.output_dir)
    
    # # Log to wandb if requested
    # if args.log_to_wandb:
    #     project_name = args.wandb_project or f"verl_grpo_{args.dataset_name}"
    #     log_to_wandb(hyperparams, project_name, args.experiment_name)
    
    print("Hyperparameter logging completed successfully!")


if __name__ == "__main__":
    main()
