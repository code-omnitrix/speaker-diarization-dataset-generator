"""
Speaker Diarization Dataset Preparation Script

This script prepares a dataset for fine-tuning speaker diarization models by:
1. Loading audio clips from parquet files
2. Randomly selecting speakers
3. Combining their audio clips with realistic gaps and limited overlaps
4. Generating both RTTM and CSV files with speaker diarization timestamps
5. Focusing on speaker embeddings with consistent speaker characteristics
"""

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import io
import soundfile as sf

# Configuration
NUM_SAMPLES_TO_GENERATE = 10  # Number of audio files to create
MIN_SPEAKERS = 2  # Minimum speakers per audio (at least 2 for diarization)
MAX_SPEAKERS = 5  # Maximum speakers per audio
CLIPS_PER_SPEAKER = 8  # Number of clips to use per speaker
TARGET_AUDIO_LENGTH = 80  # Target length in seconds
SILENCE_DURATION_RANGE = (0.3, 1.5)  # Min and max silence duration in seconds
OVERLAP_PROBABILITY = 0.15  # 15% chance of overlapping speech
OVERLAP_DURATION_RANGE = (0.5, 2.0)  # Duration of overlap in seconds
OUTPUT_DIR = "output_diarization_dataset"
SAMPLE_RATE = 16000  # Standard sample rate for audio

def load_all_datasets(dataset_folder):
    """Load all parquet files from the Datasets folder (handles any language structure)
    
    Supports folder structures like:
    - Datasets/language1/*.parquet
    - Datasets/language2/*.parquet
    - Any number of languages and parquet files
    """
    print("Loading datasets...")
    all_data = []
    
    dataset_path = Path(dataset_folder)
    
    # Check if dataset folder exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
    
    # Recursively find all parquet files
    parquet_files = list(dataset_path.rglob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_folder}")
    
    # Group by language folders for reporting
    language_stats = {}
    
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        all_data.append(df)
        
        # Track language statistics
        parent_folder = parquet_file.parent.name
        if parent_folder not in language_stats:
            language_stats[parent_folder] = 0
        language_stats[parent_folder] += len(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total parquet files: {len(parquet_files)}")
    print(f"  Total records loaded: {len(combined_df)}")
    print(f"  Languages/folders detected: {list(language_stats.keys())}")
    for lang, count in language_stats.items():
        print(f"    - {lang}: {count} records")
    
    # Check if required columns exist
    required_columns = ['audio_filepath', 'speaker_id', 'lang']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"  Unique speakers: {combined_df['speaker_id'].nunique()}")
    print(f"  Languages in data: {combined_df['lang'].unique().tolist()}\n")
    
    return combined_df

def extract_audio_from_bytes(audio_bytes):
    """Extract audio data from byte format"""
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes['bytes']))
        return audio_data, sr
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None

def generate_silence(duration, sample_rate):
    """Generate silence (near-zero values) for the specified duration"""
    num_samples = int(duration * sample_rate)
    # Very low amplitude background noise to simulate silence
    silence = np.random.normal(0, 0.001, num_samples)
    return silence

def mix_audio_with_overlap(audio1, audio2, overlap_duration, sample_rate):
    """Mix two audio clips with specified overlap duration"""
    overlap_samples = int(overlap_duration * sample_rate)
    
    if len(audio1) < overlap_samples or len(audio2) < overlap_samples:
        overlap_samples = min(len(audio1), len(audio2))
    
    # Take the end of audio1 and beginning of audio2 for overlap
    audio1_overlap = audio1[-overlap_samples:]
    audio2_overlap = audio2[:overlap_samples]
    
    # Mix the overlapping portions (simple addition with normalization)
    mixed_overlap = (audio1_overlap + audio2_overlap) * 0.6
    
    # Combine: audio1 (except overlap) + mixed + audio2 (except overlap)
    result = np.concatenate([
        audio1[:-overlap_samples],
        mixed_overlap,
        audio2[overlap_samples:]
    ])
    
    return result, overlap_samples / sample_rate

def create_diarization_audio(df, num_speakers, clips_per_speaker, target_length, sample_rate):
    """Create a single audio file with multiple speakers and return diarization info
    
    This includes:
    - Realistic silence between speakers
    - Limited overlapping speech segments
    - Consistent speaker characteristics throughout
    """
    
    # Select random speakers
    unique_speakers = df['speaker_id'].unique()
    if len(unique_speakers) < num_speakers:
        num_speakers = len(unique_speakers)
    
    selected_speakers = random.sample(list(unique_speakers), num_speakers)
    print(f"Selected {num_speakers} speakers: {selected_speakers}")
    
    # Collect audio clips for each speaker
    audio_clips_by_speaker = {}
    
    for speaker in selected_speakers:
        speaker_df = df[df['speaker_id'] == speaker]
        # Take random clips from this speaker
        num_clips = min(clips_per_speaker, len(speaker_df))
        selected_clips = speaker_df.sample(n=num_clips)
        
        clips = []
        for idx, row in selected_clips.iterrows():
            audio_data, sr = extract_audio_from_bytes(row['audio_filepath'])
            if audio_data is not None:
                # Resample if needed
                if sr != sample_rate:
                    from scipy import signal
                    num_samples = int(len(audio_data) * sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)
                
                clips.append({
                    'audio': audio_data,
                    'speaker': speaker,
                    'language': row['lang'],
                    'duration': len(audio_data) / sample_rate
                })
        
        audio_clips_by_speaker[speaker] = clips
    
    # Build the final audio file with timing information
    audio_segments = []  # List of (audio_array, start_time, end_time, speaker, language)
    diarization_records = []
    current_time = 0.0
    previous_speaker = None
    
    while current_time < target_length:
        # Randomly select a speaker (prefer different from previous)
        available_speakers = [s for s in selected_speakers if s != previous_speaker]
        if not available_speakers:
            available_speakers = selected_speakers
        speaker = random.choice(available_speakers)
        
        # Check if this speaker has clips available
        if not audio_clips_by_speaker[speaker]:
            break
        
        # Randomly select a clip from this speaker
        clip_info = random.choice(audio_clips_by_speaker[speaker])
        audio_clip = clip_info['audio']
        clip_duration = clip_info['duration']
        
        # Decide if this clip should overlap with previous
        should_overlap = (
            len(audio_segments) > 0 and 
            random.random() < OVERLAP_PROBABILITY and
            previous_speaker != speaker
        )
        
        if should_overlap:
            # Create overlapping speech
            overlap_duration = random.uniform(*OVERLAP_DURATION_RANGE)
            overlap_duration = min(overlap_duration, clip_duration * 0.5)  # Max 50% overlap
            
            # Adjust start time to create overlap
            start_time = current_time - overlap_duration
            end_time = start_time + clip_duration
            
            audio_segments.append({
                'audio': audio_clip,
                'start': start_time,
                'end': end_time,
                'speaker': speaker,
                'language': clip_info['language']
            })
            
            current_time = end_time
            
        else:
            # Normal sequential speech with silence gap
            if len(audio_segments) > 0:
                # Add silence between speakers
                silence_duration = random.uniform(*SILENCE_DURATION_RANGE)
                current_time += silence_duration
            
            start_time = current_time
            end_time = current_time + clip_duration
            
            audio_segments.append({
                'audio': audio_clip,
                'start': start_time,
                'end': end_time,
                'speaker': speaker,
                'language': clip_info['language']
            })
            
            current_time = end_time
        
        previous_speaker = speaker
    
    # Build final audio array by mixing overlapping segments
    if not audio_segments:
        return np.array([]), [], selected_speakers
    
    total_duration = max(seg['end'] for seg in audio_segments)
    total_samples = int(total_duration * sample_rate)
    final_audio = np.zeros(total_samples)
    
    # Mix all audio segments at their respective positions
    for seg in audio_segments:
        start_sample = int(seg['start'] * sample_rate)
        audio_data = seg['audio']
        end_sample = start_sample + len(audio_data)
        
        # Skip segments that start beyond the final audio length
        if start_sample >= len(final_audio):
            continue
        
        # Ensure we don't exceed array bounds
        if end_sample > len(final_audio):
            audio_data = audio_data[:len(final_audio) - start_sample]
            end_sample = len(final_audio)
        
        # Skip if no audio data remains after trimming
        if len(audio_data) == 0:
            continue
        
        # Add audio (overlaps will naturally mix)
        final_audio[start_sample:end_sample] += audio_data * 0.7
        
        # Record diarization info (only for segments that made it into the audio)
        diarization_records.append({
            'Speaker': seg['speaker'],
            'StartTS': seg['start'],
            'EndTS': min(seg['end'], total_duration),
            'Language': seg['language']
        })
    
    # Normalize to prevent clipping
    max_val = np.abs(final_audio).max()
    if max_val > 1.0:
        final_audio = final_audio / max_val * 0.95
    
    # Sort diarization records by start time
    diarization_records.sort(key=lambda x: x['StartTS'])
    
    return final_audio, diarization_records, selected_speakers

def count_overlaps(diarization_records):
    """Count the number of overlapping speech segments"""
    overlaps = 0
    for i, record1 in enumerate(diarization_records):
        for record2 in diarization_records[i+1:]:
            # Check if segments overlap
            start1, end1 = record1['StartTS'], record1['EndTS']
            start2, end2 = record2['StartTS'], record2['EndTS']
            
            if start1 < end2 and start2 < end1 and record1['Speaker'] != record2['Speaker']:
                overlaps += 1
    return overlaps

def save_rttm_file(diarization_records, filename, rttm_dir):
    """Save RTTM (Rich Transcription Time Marked) file for pyannote
    
    RTTM format: SPEAKER <file-id> <channel> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
    """
    os.makedirs(rttm_dir, exist_ok=True)
    rttm_filename = f"{filename}.rttm"
    rttm_path = os.path.join(rttm_dir, rttm_filename)
    
    with open(rttm_path, 'w') as f:
        for record in diarization_records:
            start = record['StartTS']
            end = record['EndTS']
            duration = end - start
            speaker = record['Speaker']
            
            # RTTM format line
            f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
    
    print(f"Saved RTTM: {rttm_path}")
    return rttm_path

def append_to_combined_rttm(diarization_records, filename, combined_rttm_path):
    """Append diarization records to a combined RTTM file
    
    This creates a single RTTM file with all samples - standard format for pyannote training
    """
    with open(combined_rttm_path, 'a') as f:
        for record in diarization_records:
            start = record['StartTS']
            end = record['EndTS']
            duration = end - start
            speaker = record['Speaker']
            
            # RTTM format line
            f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

def save_audio_csv_and_rttm(audio_data, diarization_records, filename, output_dir, sample_rate):
    """Save the audio file, CSV, and RTTM files in organized folder structure"""
    
    # Create organized directory structure
    audio_dir = os.path.join(output_dir, "audio")
    rttm_dir = os.path.join(output_dir, "rttm")
    csv_dir = os.path.join(output_dir, "csv")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save audio file in audio folder
    audio_filename = f"{filename}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)
    sf.write(audio_path, audio_data, sample_rate)
    print(f"Saved audio: {audio_path}")
    
    # Prepare CSV data (in the specified format)
    csv_data = []
    for idx, record in enumerate(diarization_records, start=1):
        csv_data.append({
            'AudioFileName': audio_filename,
            'Speaker': record['Speaker'],
            'StartTS': f"{record['StartTS']:.3f}",
            'EndTS': f"{record['EndTS']:.3f}",
            'Language': record['Language']
        })
    
    # Save CSV file in csv folder
    csv_filename = f"{filename}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(csv_path, index=True, index_label='')
    print(f"Saved CSV: {csv_path}")
    
    # Save RTTM file in rttm folder
    rttm_path = save_rttm_file(diarization_records, filename, rttm_dir)
    
    return audio_path, csv_path, rttm_path

def main():
    """Main function to prepare the diarization dataset"""
    
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    # np.random.seed(42)
    
    # Load datasets (works with any language folder structure)
    dataset_folder = "../Datasets"
    try:
        df = load_all_datasets(dataset_folder)
    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        print(f"\nMake sure your Datasets folder has this structure:")
        print(f"  Datasets/")
        print(f"    language1/")
        print(f"      file1.parquet")
        print(f"      file2.parquet")
        print(f"    language2/")
        print(f"      file1.parquet")
        return
    
    print(f"\n🚀 Generating {NUM_SAMPLES_TO_GENERATE} audio samples...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Store all diarization data for combined files
    all_diarization_data = []
    combined_rttm_path = os.path.join(OUTPUT_DIR, "all_samples_combined.rttm")
    
    # Clear combined RTTM file if it exists
    if os.path.exists(combined_rttm_path):
        os.remove(combined_rttm_path)
    
    for i in range(NUM_SAMPLES_TO_GENERATE):
        print(f"\n--- Generating sample {i+1}/{NUM_SAMPLES_TO_GENERATE} ---")
        
        # Random number of speakers for this sample
        num_speakers = random.randint(MIN_SPEAKERS, MAX_SPEAKERS)
        
        # Create diarization audio
        audio_data, diarization_records, speakers = create_diarization_audio(
            df, 
            num_speakers, 
            CLIPS_PER_SPEAKER, 
            TARGET_AUDIO_LENGTH, 
            SAMPLE_RATE
        )
        
        # Save audio, CSV, and RTTM
        filename = f"diarization_sample_{i+1:03d}"
        audio_path, csv_path, rttm_path = save_audio_csv_and_rttm(
            audio_data, 
            diarization_records, 
            filename, 
            OUTPUT_DIR, 
            SAMPLE_RATE
        )
        
        # Append to combined RTTM file (for pyannote training)
        append_to_combined_rttm(diarization_records, filename, combined_rttm_path)
        
        # Collect data for combined CSV
        audio_filename = f"{filename}.wav"
        for record in diarization_records:
            all_diarization_data.append({
                'AudioFileName': audio_filename,
                'Speaker': record['Speaker'],
                'StartTS': f"{record['StartTS']:.3f}",
                'EndTS': f"{record['EndTS']:.3f}",
                'Language': record['Language']
            })
        
        # Count overlaps
        num_overlaps = count_overlaps(diarization_records)
        print(f"Sample {i+1} created: {num_speakers} speakers, {len(diarization_records)} segments, {num_overlaps} overlaps")
    
    # Save combined CSV file in parent directory
    if all_diarization_data:
        combined_csv_path = os.path.join(OUTPUT_DIR, "all_samples_combined.csv")
        df_combined = pd.DataFrame(all_diarization_data)
        df_combined.to_csv(combined_csv_path, index=True, index_label='')
        print(f"\n✅ Combined CSV saved: {combined_csv_path}")
        print(f"   Total segments across all samples: {len(all_diarization_data)}")
    
    # Report combined RTTM in parent directory
    if os.path.exists(combined_rttm_path):
        with open(combined_rttm_path, 'r') as f:
            rttm_lines = len(f.readlines())
        print(f"\n✅ Combined RTTM saved: {combined_rttm_path}")
        print(f"   Total RTTM entries: {rttm_lines}")
        print(f"   Ready for pyannote.audio training!")
    
    print(f"\n✅ Dataset preparation complete! Output saved to '{OUTPUT_DIR}' folder")
    print(f"\n📁 Output Structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── all_samples_combined.csv (all labels in CSV)")
    print(f"   ├── all_samples_combined.rttm (all labels in RTTM for pyannote)")
    print(f"   ├── audio/ ({NUM_SAMPLES_TO_GENERATE} .wav files)")
    print(f"   ├── rttm/ ({NUM_SAMPLES_TO_GENERATE} .rttm files)")
    print(f"   └── csv/ ({NUM_SAMPLES_TO_GENERATE} .csv files)")

if __name__ == "__main__":
    main()
