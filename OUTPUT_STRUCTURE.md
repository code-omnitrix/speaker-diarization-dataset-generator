# 📂 Output Structure Summary

## New Organized Structure

```
output_diarization_dataset/
│
├── 📄 all_samples_combined.csv          ← Use this for general ML training
├── 📄 all_samples_combined.rttm         ← Use this for pyannote.audio training
│
├── 📁 audio/                            ← All audio files
│   ├── diarization_sample_001.wav
│   ├── diarization_sample_002.wav
│   ├── diarization_sample_003.wav
│   ├── ...
│   └── diarization_sample_010.wav
│
├── 📁 rttm/                             ← Individual RTTM files
│   ├── diarization_sample_001.rttm
│   ├── diarization_sample_002.rttm
│   ├── diarization_sample_003.rttm
│   ├── ...
│   └── diarization_sample_010.rttm
│
└── 📁 csv/                              ← Individual CSV files
    ├── diarization_sample_001.csv
    ├── diarization_sample_002.csv
    ├── diarization_sample_003.csv
    ├── ...
    └── diarization_sample_010.csv
```

## Why This Structure?

### ✅ Benefits

1. **Organized**: Files are grouped by type (audio, rttm, csv)
2. **Clean Parent Directory**: Only the important combined files at top level
3. **Easy Training**: Combined files are immediately visible
4. **Scalable**: Can generate hundreds of samples without clutter
5. **Standard Practice**: Follows common ML dataset organization patterns

### 🎯 Use Cases

| Use Case | Files to Use | Location |
|----------|--------------|----------|
| **Pyannote Training** | `all_samples_combined.rttm` | Parent directory |
| **General ML Training** | `all_samples_combined.csv` | Parent directory |
| **Audio Processing** | All `.wav` files | `audio/` folder |
| **Per-Sample Analysis** | Individual `.rttm` or `.csv` | `rttm/` or `csv/` folders |
| **Custom Pipeline** | Mix and match | All folders |

## File Contents

### Parent Directory Files

**`all_samples_combined.csv`**
- Contains: All diarization labels for all samples
- Format: AudioFileName, Speaker, StartTS, EndTS, Language
- Use for: Training general ML models, data analysis

**`all_samples_combined.rttm`**
- Contains: All diarization labels in RTTM format
- Format: Standard RTTM (pyannote compatible)
- Use for: pyannote.audio training, speaker diarization benchmarks

### Subfolder Files

**`audio/*.wav`**
- Contains: Mixed audio with multiple speakers and overlaps
- Format: 16kHz WAV files
- Use for: Audio input to models

**`rttm/*.rttm`**
- Contains: Individual RTTM annotations per audio file
- Format: Standard RTTM format
- Use for: Per-file processing, validation

**`csv/*.csv`**
- Contains: Individual CSV annotations per audio file
- Format: Same as combined CSV but per file
- Use for: Per-file analysis, custom processing

## Migration from Old Structure

If you have old outputs (all files in one folder), the new structure:
- Keeps combined files in same location (parent directory)
- Organizes individual files into subfolders
- No breaking changes for combined file usage

## Quick Access

```bash
# Access combined files (for training)
ls output_diarization_dataset/*.{csv,rttm}

# Access all audio files
ls output_diarization_dataset/audio/

# Access all RTTM files
ls output_diarization_dataset/rttm/

# Access all CSV files
ls output_diarization_dataset/csv/

# Count total files
find output_diarization_dataset -type f | wc -l
```

## For Pyannote.audio Users

The combined RTTM file location hasn't changed:
```python
rttm_path = "output_diarization_dataset/all_samples_combined.rttm"
audio_dir = "output_diarization_dataset/audio/"
```

Your training pipelines can point to these paths without changes!
