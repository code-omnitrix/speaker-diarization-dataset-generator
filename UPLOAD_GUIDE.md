# 🚀 Hugging Face Dataset Upload Guide

This guide helps you upload your synthetic multilingual speaker diarization dataset to Hugging Face Hub for easy sharing and access.

## 📋 Prerequisites

### 1. Hugging Face Account
- Create account at [huggingface.co](https://huggingface.co)
- Generate access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- Choose "Write" permissions for uploading datasets

### 2. Python Environment
- Python 3.8 or higher
- Required packages (installed automatically)

### 3. Dataset Files
Your dataset should have this structure:
```
output_diarization_dataset/
├── all_samples_combined.csv    # Main annotations file
├── audio/                      # Audio files folder
│   ├── diarization_sample_0001.wav
│   ├── diarization_sample_0002.wav
│   └── ... (all your audio files)
├── csv/                       # Individual CSV files (optional)
└── rttm/                      # Individual RTTM files (optional)
```

## 🎯 Quick Start

### Method 1: Run Batch Script (Windows)
```bash
# Double-click or run in terminal
run_upload.bat
```

### Method 2: Manual Python Execution
```bash
# Install requirements
pip install -r upload_requirements.txt

# Run upload script
python upload_to_huggingface.py
```

## 📊 What Gets Uploaded

### Dataset Information
- **Total Files**: All your audio files (626+)
- **Duration**: ~15+ hours of synthetic audio
- **Languages**: Hindi, English, Punjabi
- **Format**: HuggingFace Dataset with audio arrays + annotations
- **Splits**: Automatic 80% train / 10% validation / 10% test

### Dataset Structure (HF Format)
```python
{
    'audio': {
        'array': numpy.ndarray,  # 16kHz audio waveform
        'sampling_rate': 16000
    },
    'file_id': str,              # File identifier
    'speakers': List[str],       # ["Speaker_00", "Speaker_01", ...]
    'start_times': List[float],  # Segment start times
    'end_times': List[float],    # Segment end times
    'languages': List[str],      # ["hi", "en", "pa", ...]
    'duration': float,           # Total duration in seconds
    'num_segments': int,         # Number of speech segments
    'unique_speakers': List[str] # Unique speakers in file
}
```

## 🔧 Configuration Options

### 1. Change Dataset Name
Edit in `upload_to_huggingface.py`:
```python
DATASET_NAME = "your-custom-dataset-name"
```

### 2. Change Dataset Path
If your dataset is in a different location:
```python
DATASET_PATH = "/path/to/your/dataset"
```

### 3. Make Dataset Private
Set private repository:
```python
private=True  # In create_repo() function
```

## 📤 Upload Process Steps

The script automatically:

1. **🔐 Login**: Prompts for HuggingFace token
2. **🔍 Validate**: Checks dataset structure and files
3. **📊 Process**: Converts to HuggingFace Dataset format
4. **✂️ Split**: Creates train/val/test splits
5. **📝 Document**: Generates comprehensive dataset card
6. **🏗️ Create**: Sets up HuggingFace repository
7. **📤 Upload**: Pushes dataset and documentation

## 🎉 After Upload

### Your Dataset Will Be Available At:
```
https://huggingface.co/datasets/{your-username}/{dataset-name}
```

### Loading Your Dataset:
```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("your-username/your-dataset-name")

# Load specific split
train_data = load_dataset("your-username/your-dataset-name", split="train")

# Access sample
sample = dataset['train'][0]
audio = sample['audio']['array']
speakers = sample['speakers']
```

### Example Usage:
```python
# For pyannote.audio training
from pyannote.audio import Model

sample = dataset['train'][0]
print(f"Duration: {sample['duration']:.2f}s")
print(f"Speakers: {sample['speakers']}")
print(f"Languages: {set(sample['languages'])}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Error**
   - Ensure you have a valid HuggingFace token
   - Token needs "Write" permissions

2. **File Not Found**
   - Check `DATASET_PATH` points to correct folder
   - Ensure `all_samples_combined.csv` exists

3. **Memory Issues**
   - Large datasets may need chunked uploading
   - Consider reducing batch size in script

4. **Network Timeout**
   - Large audio files may timeout
   - Try running with stable internet connection

### Getting Help

1. Check the console output for detailed error messages
2. Verify your dataset structure matches expected format
3. Ensure all audio files are accessible and valid WAV format
4. Check HuggingFace Hub status: [status.huggingface.co](https://status.huggingface.co)

## 📚 Additional Features

### Dataset Card
Automatically generates comprehensive documentation including:
- Dataset statistics and overview
- Usage examples and code snippets
- Language and speaker information
- Technical specifications
- Citation information

### Quality Checks
- Validates audio file integrity
- Checks annotation completeness
- Reports missing files or corrupted data
- Provides detailed statistics

### Compatibility
- Ready for pyannote.audio training
- Compatible with speechbrain, ESPnet
- Standard HuggingFace Datasets format
- Cross-platform support

## 🎯 Next Steps After Upload

1. **Share Your Dataset**: Send the HuggingFace URL to colleagues
2. **Train Models**: Use for speaker diarization research
3. **Community**: Add to HuggingFace model cards that use your data
4. **Updates**: Re-run script to update with new data

---

**Happy Uploading! 🚀** Your synthetic multilingual dataset will help advance speaker diarization research!