# Dataset Structure Guide

This document explains how the script handles different dataset folder structures.

## Supported Structures

The script is **fully flexible** and automatically adapts to any folder structure containing parquet files.

### Example 1: Language-based folders (Current setup)
```
Datasets/
├── punjabi/
│   ├── punjabi1.parquet
│   ├── punjabi2.parquet
│   └── punjabi3.parquet
└── hindi/
    ├── hindi1.parquet
    ├── hindi2.parquet
    └── hindi3.parquet
```

### Example 2: Multiple languages
```
Datasets/
├── hindi/
│   └── data.parquet
├── punjabi/
│   └── data.parquet
├── english/
│   └── data.parquet
├── tamil/
│   └── data.parquet
└── bengali/
    └── data.parquet
```

### Example 3: Flat structure
```
Datasets/
├── dataset1.parquet
├── dataset2.parquet
├── dataset3.parquet
└── dataset4.parquet
```

### Example 4: Nested structure
```
Datasets/
├── region1/
│   ├── language1/
│   │   ├── file1.parquet
│   │   └── file2.parquet
│   └── language2/
│       └── file1.parquet
└── region2/
    └── language3/
        └── file1.parquet
```

## How It Works

1. **Recursive Search**: The script uses `rglob("*.parquet")` to find ALL parquet files regardless of nesting level

2. **Automatic Detection**: It automatically detects all parquet files and loads them

3. **Statistics Reporting**: Shows you:
   - Number of parquet files found
   - Total records loaded
   - Folder names detected (usually languages)
   - Records per folder
   - Unique speakers
   - Languages in the data

4. **Validation**: Checks for required columns:
   - `audio_filepath` (contains audio bytes)
   - `speaker_id` (speaker identifier)
   - `lang` (language code)

## Required Columns in Parquet Files

Your parquet files **must** contain these columns:

| Column Name | Type | Description |
|------------|------|-------------|
| `audio_filepath` | dict | Dictionary with 'bytes' key containing audio data |
| `speaker_id` | string | Unique identifier for each speaker |
| `lang` | string | Language code (e.g., 'hi', 'en', 'pa') |
| `duration` | float | Duration of audio in seconds (optional but helpful) |

## Example Output

When you run the script, you'll see:

```
Loading datasets...
Loading Datasets/punjabi/punjabi1.parquet...
Loading Datasets/punjabi/punjabi2.parquet...
Loading Datasets/hindi/hindi1.parquet...
Loading Datasets/hindi/hindi2.parquet...

📊 Dataset Statistics:
  Total parquet files: 4
  Total records loaded: 12450
  Languages/folders detected: ['punjabi', 'hindi']
    - punjabi: 5623 records
    - hindi: 6827 records
  Unique speakers: 187
  Languages in data: ['pa', 'hi']
```

## Adding New Languages

To add a new language:

1. Create a folder in `Datasets/` (e.g., `Datasets/telugu/`)
2. Add your parquet files
3. Run the script - it will automatically detect and use them!

No code changes needed! 🎉
