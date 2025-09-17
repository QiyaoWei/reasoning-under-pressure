from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "radadjoneva/func-corr-qwen2.5-1.5b-regularRL-300"
folder_path = "checkpoints/verl_grpo_function_correctness/qwen2.5-1.5b-instruct-regularRL-kl0.0/global_step_300"

upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    repo_type="model",
    commit_message="Upload checkpoint 300 for qwen2.5-1.5b-instruct-regularRL-kl0.0",
)