#!/usr/bin/env python3
"""
Download a specific subdirectory from a HuggingFace repository.
"""

import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path

def download_subdirectory(repo_id: str, subdirectory: str, local_dir: str = None):
    """
    Download a specific subdirectory from a HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'BenatCambridge/verl')
        subdirectory: Path to subdirectory within the repo
        local_dir: Local directory to save files (optional)
    """
    if local_dir is None:
        # Create a local directory based on repo name and subdirectory
        local_dir = f"./{repo_id.split('/')[-1]}_{subdirectory.replace('/', '_')}"
    
    print(f"Downloading {subdirectory} from {repo_id} to {local_dir}")
    
    # Download only the specific subdirectory
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{subdirectory}/**",  # Only download files in this subdirectory
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Download actual files, not symlinks
    )
    
    # The files will be downloaded to local_dir/subdirectory
    # Move them to the root of local_dir for easier access
    subdirectory_path = os.path.join(local_dir, subdirectory)
    if os.path.exists(subdirectory_path):
        print(f"Files downloaded to: {subdirectory_path}")
        
        # Optionally move files to parent directory
        # import shutil
        # for item in os.listdir(subdirectory_path):
        #     shutil.move(os.path.join(subdirectory_path, item), local_dir)
        # shutil.rmtree(subdirectory_path)
    
    return subdirectory_path

def main():
    parser = argparse.ArgumentParser(description="Download subdirectory from HuggingFace")
    parser.add_argument("--repo", type=str, required=True,
                       help="HuggingFace repository ID (e.g., 'BenatCambridge/verl')")
    parser.add_argument("--subdirectory", type=str, required=True,
                       help="Subdirectory path (e.g., 'verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_100/actor/huggingface')")
    parser.add_argument("--local-dir", type=str, default=None,
                       help="Local directory to save files")
    
    args = parser.parse_args()
    
    downloaded_path = download_subdirectory(
        repo_id=args.repo,
        subdirectory=args.subdirectory,
        local_dir=args.local_dir
    )
    
    print(f"\nDownload complete! Model files are in: {downloaded_path}")
    print(f"\nYou can now run inference with:")
    print(f"python run_inference_diamonds.py --checkpoint {downloaded_path}")

if __name__ == "__main__":
    main()