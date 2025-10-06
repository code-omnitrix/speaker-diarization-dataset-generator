#!/usr/bin/env python3
"""
🚀 Hugging Face Dataset Upload Script
=====================================

This script uploads your synthetic multilingual speaker diarization dataset 
to Hugging Face Hub for easy sharing and access.

Created for: Synthetic Multilingual Speaker Diarization Dataset
Languages: Hindi (hi), English (en), Punjabi (pa)
Format: Speaker_00, Speaker_01, Speaker_02 etc.

Usage:
    python upload_to_huggingface.py
    
Requirements:
    pip install huggingface_hub datasets librosa soundfile pandas
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf

# Hugging Face imports
from huggingface_hub import HfApi, Repository, login
from datasets import Dataset, DatasetDict, Audio, Features, Value, Sequence

# Configuration
DATASET_NAME = "synthetic-multilingual-speaker-diarization"  # Change this to your preferred name
HF_USERNAME = None  # Will be set during login
DATASET_PATH = "./output_diarization_dataset"  # Path to your generated dataset
REPO_TYPE = "dataset"

class HuggingFaceUploader:
    """Handles uploading synthetic diarization dataset to Hugging Face Hub"""
    
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.hf_api = HfApi()
        self.repo_id = None
        
    def login_to_hf(self):
        """Login to Hugging Face Hub"""
        print("🔐 Logging into Hugging Face Hub...")
        print("Please enter your Hugging Face token (get it from: https://huggingface.co/settings/tokens)")
        
        try:
            login()
            whoami = self.hf_api.whoami()
            global HF_USERNAME
            HF_USERNAME = whoami['name']
            self.repo_id = f"{HF_USERNAME}/{self.dataset_name}"
            print(f"✅ Successfully logged in as: {HF_USERNAME}")
            print(f"📦 Dataset will be uploaded to: {self.repo_id}")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False
    
    def validate_dataset(self):
        """Validate the dataset structure and files"""
        print("🔍 Validating dataset structure...")
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Check required files
        required_files = [
            "all_samples_combined.csv",
            "audio"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.dataset_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Load and validate CSV
        csv_path = self.dataset_path / "all_samples_combined.csv"
        df = pd.read_csv(csv_path)
        
        # Remove unnamed index column if exists
        if df.columns[0].startswith('Unnamed'):
            df = df.drop(df.columns[0], axis=1)
        
        required_columns = ['AudioFileName', 'Speaker', 'StartTS', 'EndTS', 'Language']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Check audio files
        audio_dir = self.dataset_path / "audio"
        unique_files = df['AudioFileName'].unique()
        audio_files = list(audio_dir.glob("*.wav"))
        
        print(f"✅ Dataset validation passed:")
        print(f"   📊 CSV entries: {len(df)}")
        print(f"   🎵 Unique audio files in CSV: {len(unique_files)}")
        print(f"   📁 Audio files on disk: {len(audio_files)}")
        print(f"   👥 Unique speakers: {len(df['Speaker'].unique())}")
        print(f"   🌍 Languages: {list(df['Language'].unique())}")
        
        return df, audio_dir
    
    def create_hf_dataset(self, df, audio_dir):
        """Create HuggingFace Dataset from local files"""
        print("🔄 Creating HuggingFace Dataset...")
        
        # Get unique audio files
        unique_files = df['AudioFileName'].unique()
        dataset_entries = []
        
        print(f"📊 Processing {len(unique_files)} audio files...")
        
        for i, audio_file in enumerate(unique_files):
            if (i + 1) % 50 == 0 or i == len(unique_files) - 1:
                print(f"   Processed {i + 1}/{len(unique_files)} files...")
            
            # Get annotations for this file
            file_annotations = df[df['AudioFileName'] == audio_file].copy()
            file_annotations = file_annotations.sort_values('StartTS')
            
            # Audio file path
            audio_path = audio_dir / audio_file
            if not audio_path.exists():
                print(f"⚠️ Warning: Audio file not found: {audio_path}")
                continue
            
            # Load audio
            try:
                audio_data, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"❌ Error loading {audio_path}: {e}")
                continue
            
            # Extract data
            speakers = file_annotations['Speaker'].tolist()
            start_times = file_annotations['StartTS'].tolist()
            end_times = file_annotations['EndTS'].tolist()
            languages = file_annotations['Language'].tolist()
            
            # Create entry
            entry = {
                'audio': {
                    'path': str(audio_path),
                    'array': audio_data,
                    'sampling_rate': sr
                },
                'file_id': audio_file.replace('.wav', ''),
                'speakers': speakers,
                'start_times': start_times,
                'end_times': end_times,
                'languages': languages,
                'duration': len(audio_data) / sr,
                'num_segments': len(speakers),
                'unique_speakers': list(set(speakers))
            }
            
            dataset_entries.append(entry)
        
        print(f"✅ Successfully processed {len(dataset_entries)} audio files")
        
        # Create Dataset
        dataset = Dataset.from_list(dataset_entries)
        
        # Cast audio column
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def create_dataset_splits(self, dataset):
        """Create train/validation/test splits"""
        print("📊 Creating dataset splits...")
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        
        total_samples = len(dataset)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        
        # Create splits
        train_dataset = dataset.select(range(0, train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_samples))
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        print(f"✅ Dataset splits created:")
        print(f"   🏋️ Train: {len(train_dataset)} samples")
        print(f"   🔍 Validation: {len(val_dataset)} samples")
        print(f"   🧪 Test: {len(test_dataset)} samples")
        
        return dataset_dict
    
    def create_dataset_card(self, dataset_dict):
        """Create a comprehensive dataset card"""
        print("📝 Creating dataset card...")
        
        # Calculate statistics
        total_samples = sum(len(split) for split in dataset_dict.values())
        total_duration = sum(sample['duration'] for split in dataset_dict.values() for sample in split)
        
        # Get sample for analysis
        sample = dataset_dict['train'][0]
        all_speakers = set()
        all_languages = set()
        
        for split in dataset_dict.values():
            for sample in split:
                all_speakers.update(sample['speakers'])
                all_languages.update(sample['languages'])
        
        dataset_card = f"""---
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

# Synthetic Multilingual Speaker Diarization Dataset

## 📊 Dataset Overview

This is a synthetic multilingual speaker diarization dataset containing **{total_samples} audio samples** with **{total_duration/3600:.2f} hours** of audio across **{len(all_languages)} languages**.

### 🎯 Key Features

- **📈 Total Samples**: {total_samples:,} audio files
- **⏰ Total Duration**: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)  
- **🌍 Languages**: {', '.join(sorted(all_languages))} (Hindi, English, Punjabi)
- **👥 Speaker Format**: {', '.join(sorted(list(all_speakers))[:10])}... (Speaker_00, Speaker_01, etc.)
- **🎵 Audio Quality**: 16kHz sampling rate, WAV format
- **📊 Splits**: 80% train, 10% validation, 10% test

### 🎙️ Dataset Statistics

| Split | Samples | Duration (hours) |
|-------|---------|------------------|
| Train | {len(dataset_dict['train']):,} | {sum(s['duration'] for s in dataset_dict['train'])/3600:.2f} |
| Validation | {len(dataset_dict['validation']):,} | {sum(s['duration'] for s in dataset_dict['validation'])/3600:.2f} |
| Test | {len(dataset_dict['test']):,} | {sum(s['duration'] for s in dataset_dict['test'])/3600:.2f} |

## 🏗️ Dataset Structure

Each sample contains:

```python
{{
    'audio': {{
        'array': numpy.ndarray,  # Audio waveform at 16kHz
        'sampling_rate': 16000
    }},
    'file_id': str,              # Unique file identifier
    'speakers': List[str],       # Speaker labels (Speaker_00, Speaker_01, etc.)
    'start_times': List[float],  # Segment start times in seconds
    'end_times': List[float],    # Segment end times in seconds  
    'languages': List[str],      # Language codes (hi, en, pa)
    'duration': float,           # Total audio duration in seconds
    'num_segments': int,         # Number of speech segments
    'unique_speakers': List[str] # Unique speakers in this file
}}
```

## 🚀 Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("{self.repo_id}")

# Load specific split
train_data = load_dataset("{self.repo_id}", split="train")
```

### Example Usage with Pyannote

```python
import librosa
from pyannote.audio import Model

# Load a sample
sample = dataset['train'][0]
audio = sample['audio']['array']
speakers = sample['speakers']
start_times = sample['start_times']
end_times = sample['end_times']

# Use for training/inference
print(f"Audio shape: {{audio.shape}}")
print(f"Speakers: {{speakers}}")
print(f"Duration: {{sample['duration']:.2f}}s")
```

## 🎯 Use Cases

- **Speaker Diarization**: Train models to identify "who spoke when"
- **Multilingual ASR**: Develop speech recognition for Hindi, English, Punjabi
- **Code-switching Research**: Study language switching patterns
- **Speaker Recognition**: Identify individual speakers across languages
- **Audio Segmentation**: Temporal speech boundary detection

## 📋 Languages & Speakers

### Languages Supported
- **Hindi (hi)**: Primary language for most segments
- **English (en)**: Code-switching and standalone segments  
- **Punjabi (pa)**: Regional language segments

### Speaker Format
- Consistent naming: `Speaker_00`, `Speaker_01`, `Speaker_02`
- Multiple speakers per audio file
- Realistic conversation patterns with speaker switching

## 🛠️ Technical Details

- **Sample Rate**: 16 kHz (compatible with most speech models)
- **Audio Format**: WAV files with float32 arrays
- **Annotation Format**: Precise timestamps with speaker and language labels
- **Quality**: Synthetic generation ensures clean, consistent audio
- **Compatibility**: Ready for use with pyannote.audio, speechbrain, ESPnet

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@dataset{{synthetic_multilingual_diarization_2025,
    title={{Synthetic Multilingual Speaker Diarization Dataset}},
    author={{Generated using synthetic data generation pipeline}},
    year={{2025}},
    url={{https://huggingface.co/datasets/{self.repo_id}}}
}}
```

## 📄 License

This dataset is released under the MIT License. See LICENSE for details.

## 🔧 Generation Details

This dataset was created using a synthetic data generation pipeline that:
- Combines speech samples from multiple speakers and languages
- Creates realistic conversation scenarios with speaker switching
- Generates precise temporal annotations for each speech segment
- Ensures balanced representation across languages and speakers

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return dataset_card
    
    def create_repository(self):
        """Create Hugging Face repository"""
        print(f"🏗️ Creating repository: {self.repo_id}")
        
        try:
            self.hf_api.create_repo(
                repo_id=self.repo_id,
                repo_type=REPO_TYPE,
                exist_ok=True,
                private=False  # Set to True if you want a private dataset
            )
            print(f"✅ Repository created/updated: https://huggingface.co/datasets/{self.repo_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return False
    
    def upload_dataset(self, dataset_dict, dataset_card):
        """Upload dataset to Hugging Face Hub"""
        print(f"📤 Uploading dataset to {self.repo_id}...")
        
        try:
            # Upload the dataset
            dataset_dict.push_to_hub(
                self.repo_id,
                commit_message="Upload synthetic multilingual speaker diarization dataset"
            )
            
            # Upload dataset card
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(dataset_card)
            
            self.hf_api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type=REPO_TYPE,
                commit_message="Add dataset card"
            )
            
            # Clean up temporary file
            os.remove("README.md")
            
            print(f"🎉 Dataset uploaded successfully!")
            print(f"🔗 Dataset URL: https://huggingface.co/datasets/{self.repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def run_upload(self):
        """Run the complete upload process"""
        print("🚀 Starting Hugging Face dataset upload process...")
        print("=" * 60)
        
        try:
            # Step 1: Login
            if not self.login_to_hf():
                return False
            
            # Step 2: Validate dataset
            df, audio_dir = self.validate_dataset()
            
            # Step 3: Create HF dataset
            dataset = self.create_hf_dataset(df, audio_dir)
            
            # Step 4: Create splits
            dataset_dict = self.create_dataset_splits(dataset)
            
            # Step 5: Create dataset card
            dataset_card = self.create_dataset_card(dataset_dict)
            
            # Step 6: Create repository
            if not self.create_repository():
                return False
            
            # Step 7: Upload
            if not self.upload_dataset(dataset_dict, dataset_card):
                return False
            
            print("=" * 60)
            print("🎉 SUCCESS! Your dataset is now available on Hugging Face Hub!")
            print(f"🔗 Dataset URL: https://huggingface.co/datasets/{self.repo_id}")
            print(f"📚 Load with: load_dataset('{self.repo_id}')")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload process failed: {e}")
            return False


def main():
    """Main function to run the upload process"""
    print("🎙️ Synthetic Multilingual Speaker Diarization Dataset Uploader")
    print("=" * 70)
    
    # Use local variables instead of global
    dataset_path = DATASET_PATH
    dataset_name = DATASET_NAME
    
    # Configuration
    print("📋 Configuration:")
    print(f"   📂 Dataset path: {dataset_path}")
    print(f"   📦 Dataset name: {dataset_name}")
    print(f"   🌍 Visibility: Public")
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to proceed with the upload? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Upload cancelled.")
        return
    
    # Optional: Change dataset name
    custom_name = input(f"\n📝 Enter custom dataset name (or press Enter for '{dataset_name}'): ").strip()
    if custom_name:
        dataset_name = custom_name
    
    # Create uploader and run
    uploader = HuggingFaceUploader(dataset_path, dataset_name)
    success = uploader.run_upload()
    
    if success:
        print("\n🎉 All done! Your dataset is ready for the community to use!")
    else:
        print("\n❌ Upload failed. Please check the errors above and try again.")


if __name__ == "__main__":
    main()