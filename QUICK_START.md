# 🚀 Quick Start Guide

## TL;DR

```bash
# 1. Check your dataset
python check_dataset.py

# 2. Generate diarization dataset
python prepare_diarization_dataset.py

# 3. Find your outputs in output_diarization_dataset/
```

## 📁 What You Get - Organized Structure

```
output_diarization_dataset/
├── all_samples_combined.csv    ← Use for general ML models
├── all_samples_combined.rttm   ← Use for pyannote.audio
├── audio/                      ← All .wav files (10 files)
├── rttm/                       ← Individual RTTM files (10 files)
└── csv/                        ← Individual CSV files (10 files)
```

### For Pyannote.audio Training:
Use `all_samples_combined.rttm` (in parent folder)

### For Other ML Models:
Use `all_samples_combined.csv` (in parent folder)

### Individual Files (if needed):
- `audio/` folder - All audio files
- `rttm/` folder - Individual RTTM per sample
- `csv/` folder - Individual CSV per sample

## Dataset Requirements

Your `Datasets/` folder must contain `.parquet` files with these columns:
- `audio_filepath` (with 'bytes' key)
- `speaker_id`
- `lang`

**Any folder structure works!** Examples:
```
Datasets/hindi/*.parquet
Datasets/punjabi/*.parquet
Datasets/tamil/*.parquet
```

## Common Questions

**Q: Can I add more languages?**  
A: Yes! Just add a new folder with parquet files. No code changes needed.

**Q: How many parquet files can I have?**  
A: Unlimited! The script loads them all automatically.

**Q: Does pyannote work with the combined RTTM?**  
A: Yes! Combined RTTM is the **standard format** for pyannote training.

**Q: Can I use this for models other than pyannote?**  
A: Yes! Use the CSV files for any ML model.

**Q: Why are files in separate folders?**  
A: For better organization! Combined files (for training) are in parent directory, individual files are organized in subfolders.

**Q: How do I adjust overlap percentage?**  
A: Edit `OVERLAP_PROBABILITY` in the script (default: 0.15 = 15%)

## Example Output

```
📊 Dataset Statistics:
  Total parquet files: 6
  Total records: 12000
  Languages: ['punjabi', 'hindi']
  Unique speakers: 150

🚀 Generating 10 audio samples...

✅ Combined CSV saved: all_samples_combined.csv
   Total segments: 247

✅ Combined RTTM saved: all_samples_combined.rttm
   Total RTTM entries: 247
   Ready for pyannote.audio training!

✅ Dataset preparation complete!

📁 Output Structure:
   output_diarization_dataset/
   ├── all_samples_combined.csv (all labels in CSV)
   ├── all_samples_combined.rttm (all labels in RTTM for pyannote)
   ├── audio/ (10 .wav files)
   ├── rttm/ (10 .rttm files)
   └── csv/ (10 .csv files)
```

## Next Steps

1. **For pyannote.audio**: Use `output_diarization_dataset/all_samples_combined.rttm`
2. **For custom models**: Use `output_diarization_dataset/all_samples_combined.csv`
3. **Audio files**: Find all in `output_diarization_dataset/audio/`
4. **Adjust parameters**: Edit configuration at top of script if needed
5. **Scale up**: Change `NUM_SAMPLES_TO_GENERATE` to create more samples

## Need Help?

- Read `README.md` for detailed documentation
- Check `DATASET_STRUCTURE.md` for folder structure examples
- See `WHATS_NEW.md` for feature details
