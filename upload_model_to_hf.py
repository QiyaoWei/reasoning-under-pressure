import argparse
import subprocess
import sys
from huggingface_hub import HfApi, upload_folder

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--repo-id", required=True, help="Repository ID (e.g., username/repo-name)")
    parser.add_argument("--folder-path", required=True, help="Path to the folder to upload")
    
    args = parser.parse_args()
    
    # Create repository using huggingface-cli
    print(f"Creating repository: {args.repo_id}")
    try:
        subprocess.run([
            "huggingface-cli", "repo", "create", 
            args.repo_id, 
            "--repo-type", "model"
        ], check=True)
        print(f"Repository {args.repo_id} created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
    
    # Extract repo name from repo_id (remove username part)
    repo_name = args.repo_id.split("/")[-1]
    commit_message = f"Upload {repo_name}"
    
    # Upload folder
    print(f"Uploading folder: {args.folder_path}")
    print(f"Commit message: {commit_message}")
    
    try:
        upload_folder(
            repo_id=args.repo_id,
            folder_path=args.folder_path,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Successfully uploaded to {args.repo_id}")
    except Exception as e:
        print(f"Error uploading folder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()