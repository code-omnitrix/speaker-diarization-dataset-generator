# Speaker Diarization Dataset Preparation

This folder contains code to prepare a dataset for fine-tuning speaker diarization models, specifically optimized for **speaker embeddings** and compatible with **pyannote.audio** and other ML frameworks.

## What it does

The script creates realistic multi-speaker audio files by:
1. **Automatically loading** audio clips from any language folder structure in `Datasets` folder
2. Randomly selecting 2-5 speakers per audio file
3. Combining their voice clips with realistic silence gaps
4. Adding **limited overlapping speech** (configurable, ~15% probability)
5. Generating **both RTTM and CSV** files with speaker diarization timestamps
6. Creating **combined RTTM file** for direct pyannote.audio training
7. Maintaining **consistent speaker characteristics** throughout (same speaker = same voice)

## 🌍 Multi-Language Support

The script automatically handles any folder structure:
```
Datasets/
  ├── hindi/
  │   ├── hindi1.parquet
  │   ├── hindi2.parquet
  │   └── hindi3.parquet
  ├── punjabi/
  │   ├── punjabi1.parquet
  │   └── punjabi2.parquet
  ├── english/
  │   └── english_data.parquet
  └── any_other_language/
      └── data.parquet
```

✅ **Works with any number of languages**  
✅ **Works with any number of parquet files**  
✅ **Automatically detects and reports language statistics**

## Key Features

✅ **Speaker Embedding Focus** - Consistent speaker voices across segments  
✅ **Overlapping Speech** - Limited realistic overlaps between speakers  
✅ **RTTM Format** - Compatible with pyannote.audio training  
✅ **CSV Format** - Custom format for other ML models  
✅ **Realistic Gaps** - Natural silence between speech segments  
✅ **Multi-language Support** - Preserves language metadata per segment  

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python prepare_diarization_dataset.py
```

## Configuration

You can modify these parameters in the script:

- `NUM_SAMPLES_TO_GENERATE = 10` - Number of audio files to create
- `MIN_SPEAKERS = 2` - Minimum speakers per audio (at least 2 for diarization)
- `MAX_SPEAKERS = 5` - Maximum speakers per audio
- `CLIPS_PER_SPEAKER = 8` - Number of clips to use per speaker
- `TARGET_AUDIO_LENGTH = 60` - Target audio length in seconds
- `SILENCE_DURATION_RANGE = (0.3, 1.5)` - Duration of silence between speakers (in seconds)
- `OVERLAP_PROBABILITY = 0.15` - Probability of overlapping speech (15% = limited overlaps)
- `OVERLAP_DURATION_RANGE = (0.5, 2.0)` - Duration of overlap when it occurs (in seconds)
- `SAMPLE_RATE = 16000` - Audio sample rate

## Output

The script creates an organized `output_diarization_dataset` folder structure:

```
output_diarization_dataset/
├── all_samples_combined.csv        # Combined CSV (all samples)
├── all_samples_combined.rttm       # Combined RTTM (all samples, for pyannote)
├── audio/                          # Audio files folder
│   ├── diarization_sample_001.wav
│   ├── diarization_sample_002.wav
│   └── ...
├── rttm/                           # RTTM files folder
│   ├── diarization_sample_001.rttm
│   ├── diarization_sample_002.rttm
│   └── ...
└── csv/                            # CSV files folder
    ├── diarization_sample_001.csv
    ├── diarization_sample_002.csv
    └── ...
```

### Parent Directory (for training):
- `all_samples_combined.csv` - All diarization data in one CSV (for general ML models)
- `all_samples_combined.rttm` - All diarization data in one RTTM (for pyannote.audio training)

### Organized Subfolders:
- `audio/` - All audio files (.wav)
- `rttm/` - Individual RTTM files per sample
- `csv/` - Individual CSV files per sample

## File Formats

### CSV Format
```
,AudioFileName,Speaker,StartTS,EndTS,Language
1,diarization_sample_001.wav,Speaker1,0.000,2.341,hi
2,diarization_sample_001.wav,Speaker2,3.500,7.552,hi
3,diarization_sample_001.wav,Speaker1,7.552,8.027,hi
...
```

Columns:
- `AudioFileName` - Name of the audio file
- `Speaker` - Speaker ID
- `StartTS` - Start timestamp (seconds, 3 decimal places)
- `EndTS` - End timestamp (seconds, 3 decimal places)
- `Language` - Language of the audio segment

### RTTM Format (for pyannote.audio)
```
SPEAKER diarization_sample_001 1 0.000 2.341 <NA> <NA> Speaker1 <NA> <NA>
SPEAKER diarization_sample_001 1 3.500 4.052 <NA> <NA> Speaker2 <NA> <NA>
SPEAKER diarization_sample_001 1 7.552 0.475 <NA> <NA> Speaker1 <NA> <NA>
...
```

Format: `SPEAKER <file-id> <channel> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>`

## Using with pyannote.audio

The generated RTTM files can be directly used with pyannote.audio's training pipeline.

### Option 1: Use Combined RTTM (Recommended)
The `all_samples_combined.rttm` file contains all samples in standard RTTM format:
```python
from pyannote.database import FileFinder, get_protocol
from pyannote.audio.tasks import SpeakerDiarization

# Use the combined RTTM file for training
# This is the standard approach - one RTTM file with all annotations
```

### Option 2: Use Individual RTTM files
Each sample also has its own `.rttm` file for separate processing

## Notes

- Overlapping speech segments are clearly visible in both formats
- Speaker IDs are preserved from the original dataset
- Audio mixing uses simple addition with normalization to prevent clipping
- The same speaker maintains consistent voice characteristics across all segments
