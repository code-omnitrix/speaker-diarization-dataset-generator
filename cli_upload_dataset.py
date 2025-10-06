#!/usr/bin/env python3
"""
Command-line dataset upload script for HuggingFace Hub
Memory-efficient alternative to the Python upload script
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
import tempfile

class CLIDatasetUploader:
    def __init__(self, token=None):
        """Initialize the CLI uploader"""
        self.token = token
        self.api = None
        
    def authenticate(self):
        """Handle authentication with HuggingFace Hub"""
        print("🔐 Authenticating with HuggingFace Hub...")
        
        if self.token:
            print("Using provided token...")
            try:
                self.api = HfApi(token=self.token)
                user_info = self.api.whoami()
                print(f"✅ Authenticated as: {user_info['name']}")
                return True
            except Exception as e:
                print(f"❌ Token authentication failed: {e}")
                return False
        else:
            # Try to use stored token or prompt for new one
            try:
                print("Please enter your HuggingFace token:")
                print("(Get it from: https://huggingface.co/settings/tokens)")
                token_input = input("Token: ").strip()
                
                if not token_input:
                    print("❌ No token provided")
                    return False
                
                self.api = HfApi(token=token_input)
                user_info = self.api.whoami()
                print(f"✅ Authenticated as: {user_info['name']}")
                
                # Save token for git credential
                self.token = token_input
                return True
                
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                return False
    
    def create_repository(self, repo_name, private=False):
        """Create HuggingFace repository"""
        try:
            print(f"🏗️ Creating repository: {repo_name}")
            
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.token,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            
            print(f"✅ Repository created/exists: {repo_url}")
            return repo_url
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return None
    
    def prepare_upload_directory(self, dataset_dir, temp_dir):
        """Prepare dataset for upload using Git LFS"""
        print("📦 Preparing dataset for upload...")
        
        # Copy dataset to temp directory
        dataset_path = Path(dataset_dir)
        upload_path = Path(temp_dir) / "upload"
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print("📁 Copying dataset files...")
        shutil.copytree(dataset_path / "audio", upload_path / "audio", dirs_exist_ok=True)
        shutil.copytree(dataset_path / "csv", upload_path / "csv", dirs_exist_ok=True)
        shutil.copytree(dataset_path / "rttm", upload_path / "rttm", dirs_exist_ok=True)
        
        # Copy combined files
        for file in ["all_samples_combined.csv", "all_samples_combined.rttm"]:
            src = dataset_path / file
            if src.exists():
                shutil.copy2(src, upload_path / file)
        
        return upload_path
    
    def create_dataset_card(self, upload_path):
        """Create README.md with dataset card"""
        print("📝 Creating dataset card...")
        
        readme_content = '''---
license: mit
task_categories:
- automatic-speech-recognition
- audio-classification
language:
- hi
- en
- pa
tags:
- speaker-diarization
- multilingual
- synthetic-data
- audio
- speech
- hindi
- english
- punjabi
size_categories:
- 1K<n<10K
---

# Multilingual Speaker Diarization Dataset

This dataset contains synthetic multilingual speaker diarization data with Hindi, English, and Punjabi audio samples.

## Dataset Structure

```
├── audio/           # WAV audio files (16kHz)
├── csv/            # Individual CSV annotations
├── rttm/           # RTTM format files for diarization
├── all_samples_combined.csv  # Complete dataset annotations
└── all_samples_combined.rttm # Complete RTTM annotations
```

## Statistics

- **Total samples**: 627 audio files
- **Total duration**: ~15 hours
- **Languages**: Hindi, English, Punjabi (monolingual, bilingual, and trilingual conversations)
- **Speaker count**: 2-5 speakers per conversation
- **Noise levels**: Clean, low, medium, high noise conditions
- **Sample rate**: 16kHz
- **Format**: WAV audio files with CSV/RTTM annotations

## Language Distribution

- Hindi only: 83 samples (13.2%)
- Punjabi only: 99 samples (15.8%)
- English only: 64 samples (10.2%)
- Bilingual: 189 samples (30.1%)
- Trilingual: 192 samples (30.6%)

## Usage

This dataset is designed for training and evaluating speaker diarization models, particularly for multilingual scenarios.

## Citation

```bibtex
@dataset{multilingual_speaker_diarization_2025,
  title={Multilingual Speaker Diarization Dataset},
  author={Audio Agnostic Team},
  year={2025},
  url={https://huggingface.co/datasets/your-username/multilingual-speaker-diarization}
}
```
'''
        
        readme_path = upload_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ Created README.md at {readme_path}")
    
    def setup_git_lfs(self, upload_path):
        """Setup Git LFS for large files"""
        print("🔧 Setting up Git LFS...")
        
        os.chdir(upload_path)
        
        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
        
        # Track large files with LFS
        with open(".gitattributes", "w") as f:
            f.write("*.wav filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.rttm filter=lfs diff=lfs merge=lfs -text\n")
            f.write("all_samples_combined.* filter=lfs diff=lfs merge=lfs -text\n")
        
        print("✅ Git LFS configured")
    
    def upload_with_git(self, upload_path, repo_name):
        """Upload using git commands (memory efficient)"""
        print("🚀 Uploading dataset using git...")
        
        os.chdir(upload_path)
        
        # Configure git user (required for commits)
        subprocess.run(["git", "config", "user.name", "Dataset Uploader"], check=True)
        subprocess.run(["git", "config", "user.email", "uploader@huggingface.co"], check=True)
        
        # Add HuggingFace remote
        repo_url = f"https://huggingface.co/datasets/{repo_name}"
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        
        # Configure git credentials with token
        if self.token:
            # Use token authentication
            auth_url = repo_url.replace("https://", f"https://oauth2:{self.token}@")
            subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
        
        # Add files
        subprocess.run(["git", "add", "."], check=True)
        
        # Create initial commit
        subprocess.run([
            "git", "commit", "-m", "Add multilingual speaker diarization dataset"
        ], check=True)
        
        # Create main branch and push
        print("⬆️ Pushing to HuggingFace Hub...")
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        
        print("✅ Upload completed!")
    
    def upload_dataset(self, dataset_dir, repo_name, private=False):
        """Main upload function"""
        try:
            # Authenticate
            if not self.authenticate():
                return False
            
            # Create repository
            if not self.create_repository(repo_name, private):
                return False
            
            # Use temporary directory for upload preparation
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"📂 Using temporary directory: {temp_dir}")
                
                # Prepare upload directory
                upload_path = self.prepare_upload_directory(dataset_dir, temp_dir)
                
                # Create dataset card
                self.create_dataset_card(upload_path)
                
                # Setup Git LFS
                self.setup_git_lfs(upload_path)
                
                # Upload using git
                self.upload_with_git(upload_path, repo_name)
            
            print("🎉 Dataset uploaded successfully!")
            print(f"📍 View at: https://huggingface.co/datasets/{repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--dataset_dir", 
                       default="output_diarization_dataset",
                       help="Dataset directory to upload")
    parser.add_argument("--repo_name", 
                       required=True,
                       help="HuggingFace repository name (username/dataset-name)")
    parser.add_argument("--token",
                       help="HuggingFace API token")
    parser.add_argument("--private", 
                       action="store_true",
                       help="Make repository private")
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not Path(args.dataset_dir).exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return
    
    print("🚀 CLI Dataset Upload to HuggingFace Hub")
    print("=" * 50)
    print(f"📂 Dataset: {args.dataset_dir}")
    print(f"🏷️  Repository: {args.repo_name}")
    print(f"🔒 Private: {args.private}")
    print("=" * 50)
    
    uploader = CLIDatasetUploader(token=args.token)
    success = uploader.upload_dataset(args.dataset_dir, args.repo_name, args.private)
    
    if not success:
        print("❌ Upload failed. Please check the logs above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())