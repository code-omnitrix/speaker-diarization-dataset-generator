# 🎉 What's New - Generalized & Enhanced Version

## ✨ Major Improvements

### 1. **Fully Generalized Dataset Loading** 🌍
- ✅ Works with **ANY number of languages**
- ✅ Works with **ANY number of parquet files**
- ✅ Handles **ANY folder structure** (nested, flat, anything)
- ✅ Automatically detects and reports language statistics
- ✅ No code changes needed when adding new languages

### 2. **Combined RTTM File for Pyannote** 🎯
- ✅ Generates `all_samples_combined.rttm` - single RTTM with all samples
- ✅ **Standard format** for pyannote.audio training
- ✅ Pyannote works BEST with combined RTTM files
- ✅ Still generates individual RTTM files per sample

### 3. **Enhanced Error Handling** 🛡️
- ✅ Clear error messages if dataset folder not found
- ✅ Validates required columns exist
- ✅ Shows helpful structure examples if loading fails
- ✅ Better progress reporting

### 4. **Better Statistics & Reporting** 📊
- ✅ Shows files loaded per language/folder
- ✅ Reports unique speakers count
- ✅ Lists all languages detected
- ✅ Shows total segments and overlaps

## 📦 Output Files

### Organized Folder Structure:
```
output_diarization_dataset/
├── all_samples_combined.csv        # All labels in one CSV (parent dir)
├── all_samples_combined.rttm       # All labels in one RTTM for pyannote (parent dir)
├── audio/                          # All audio files
│   ├── diarization_sample_001.wav
│   ├── diarization_sample_002.wav
│   └── ...
├── rttm/                           # All individual RTTM files
│   ├── diarization_sample_001.rttm
│   ├── diarization_sample_002.rttm
│   └── ...
└── csv/                            # All individual CSV files
    ├── diarization_sample_001.csv
    ├── diarization_sample_002.csv
    └── ...
```

### Key Files:
- **Parent directory**: Combined CSV and RTTM (for training)
- **audio/**: All .wav files organized together
- **rttm/**: Individual RTTM files (if needed)
- **csv/**: Individual CSV files (if needed)

## 🚀 Usage

### Step 1: Check your dataset structure
```bash
python check_dataset.py
```

This will verify your Datasets folder is properly set up.

### Step 2: Run the main script
```bash
python prepare_diarization_dataset.py
```

You'll see output like:
```
Loading datasets...
Loading Datasets/punjabi/punjabi1.parquet...
Loading Datasets/hindi/hindi1.parquet...

📊 Dataset Statistics:
  Total parquet files: 6
  Total records loaded: 12000
  Languages/folders detected: ['punjabi', 'hindi']
    - punjabi: 5000 records
    - hindi: 7000 records
  Unique speakers: 150
  Languages in data: ['pa', 'hi']

🚀 Generating 10 audio samples...

--- Generating sample 1/10 ---
Selected 3 speakers: ['S4256066400337654', 'S4257868500395463', 'S4259796200349632']
...
```

## 🎯 For Pyannote.audio Training

Use the combined RTTM file:

```python
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization

# Point to your combined RTTM file
rttm_file = "output_diarization_dataset/all_samples_combined.rttm"

# Use with pyannote training pipeline
# The combined RTTM format is the standard approach
```

## 🔧 Configuration

All parameters are at the top of `prepare_diarization_dataset.py`:

```python
NUM_SAMPLES_TO_GENERATE = 10      # Number of audio files to create
MIN_SPEAKERS = 2                  # Min speakers per audio
MAX_SPEAKERS = 5                  # Max speakers per audio
CLIPS_PER_SPEAKER = 8             # Clips per speaker
TARGET_AUDIO_LENGTH = 80          # Target length in seconds
OVERLAP_PROBABILITY = 0.15        # 15% chance of overlaps
```

## 🌟 Key Features

✅ **Multi-language**: Automatically handles any language structure  
✅ **Overlapping speech**: Limited, realistic overlaps (~15%)  
✅ **Speaker embeddings**: Consistent speaker voices throughout  
✅ **Dual format**: Both CSV and RTTM outputs  
✅ **Combined files**: Single files for all samples  
✅ **Pyannote ready**: Direct compatibility with pyannote.audio  
✅ **Flexible**: Works with custom ML models too  

## 📝 Notes

- The combined RTTM file (`all_samples_combined.rttm`) is the **recommended format** for pyannote.audio
- Individual RTTM files are still created if you need per-sample processing
- The combined CSV is useful for other ML frameworks that expect tabular data
- Language information is preserved in both formats

## 🆘 Troubleshooting

If you see errors:

1. **"Dataset folder not found"**: Check the path in the script (default: `../Datasets`)
2. **"No parquet files found"**: Make sure you have `.parquet` files in the Datasets folder
3. **"Missing required columns"**: Your parquet files need: `audio_filepath`, `speaker_id`, `lang`

Run `python check_dataset.py` to verify your setup!
