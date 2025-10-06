#!/usr/bin/env python3
"""
Simple HuggingFace dataset upload script using Hub API directly
Memory-efficient approach without temporary directories
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo, upload_folder
import argparse

def create_dataset_card():
    """Create README.md content"""
    return '''---
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
  url={https://huggingface.co/datasets/noty7gian/synthetic-multilingual-speaker-diarization}
}
```
'''

def upload_dataset_simple(dataset_dir, repo_name, token=None):
    """Simple upload using HuggingFace Hub API"""
    
    print("🚀 Simple HuggingFace Dataset Upload")
    print("=" * 50)
    
    # Authenticate
    print("🔐 Authenticating with HuggingFace Hub...")
    if token:
        api = HfApi(token=token)
    else:
        # Prompt for token
        print("Please enter your HuggingFace token:")
        print("(Get it from: https://huggingface.co/settings/tokens)")
        token = input("Token: ").strip()
        api = HfApi(token=token)
    
    # Verify authentication
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Create repository
    try:
        print(f"🏗️ Creating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            token=token,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository ready: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False
    
    # Create README.md
    dataset_path = Path(dataset_dir)
    readme_path = dataset_path / "README.md"
    
    print("📝 Creating README.md...")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(create_dataset_card())
    
    # Upload entire folder
    try:
        print("📤 Uploading dataset folder...")
        print("⚠️ This may take a while for large audio files...")
        
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            commit_message="Add multilingual speaker diarization dataset"
        )
        
        print("✅ Upload completed successfully!")
        print(f"🎉 Dataset is now available at: https://huggingface.co/datasets/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple HuggingFace dataset upload")
    parser.add_argument("--dataset_dir", 
                       default="output_diarization_dataset",
                       help="Dataset directory to upload")
    parser.add_argument("--repo_name", 
                       required=True,
                       help="HuggingFace repository name (username/dataset-name)")
    parser.add_argument("--token",
                       help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not Path(args.dataset_dir).exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return 1
    
    # Upload dataset
    success = upload_dataset_simple(args.dataset_dir, args.repo_name, args.token)
    
    if success:
        print("\n🎉 Success! Your dataset is now live on HuggingFace Hub!")
        return 0
    else:
        print("\n❌ Upload failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())