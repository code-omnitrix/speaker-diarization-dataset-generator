"""
Enhanced Speaker Diarization Dataset Preparation Script

This script prepares a multilingual synthetic fine-tuning dataset for PyAnnote speaker diarization model
with specific focus on minimizing Diarization Error Rate (DER) on Indian languages and noisy environments.

Features implemented according to dataset composition chart:
1. Audio Duration Categories: Short (30%), Medium (50%), Long (20%)
2. Speaker Count Distribution: 1 speaker (10%), 2-3 speakers (50%), 4+ speakers (40%)
3. Noise Level Categories: Clean (20%), Low (30%), Medium (30%), High (20%) with SNR levels
4. Language Composition: Hindi (15%), Punjabi (15%), English (10%), Bilingual (30%), Trilingual (30%)
5. Addresses speaker audio chunk bias by ensuring diverse clip usage
6. Implements speaker overlaps (10-20% in multi-speaker files)
7. Supports code-switching and language mixing within single files
"""

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import io
import soundfile as sf
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse

# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Using basic audio processing.")
    LIBROSA_AVAILABLE = False

try:
    from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    print("Warning: audiomentations not available. Using basic noise addition.")
    AUDIOMENTATIONS_AVAILABLE = False

# Configuration Classes
class DurationCategory(Enum):
    SHORT = "short"    # 10-60 seconds
    MEDIUM = "medium"  # 1-5 minutes
    LONG = "long"      # 5+ minutes

class NoiseLevel(Enum):
    CLEAN = "clean"        # No noise
    LOW = "low"           # SNR 15-20 dB
    MEDIUM = "medium"     # SNR 5-15 dB  
    HIGH = "high"         # SNR 0-5 dB

class LanguageComposition(Enum):
    HINDI_ONLY = "hindi_only"
    PUNJABI_ONLY = "punjabi_only"
    ENGLISH_ONLY = "english_only"
    BILINGUAL = "bilingual"
    TRILINGUAL = "trilingual"

@dataclass
class DatasetConfiguration:
    """Dataset generation configuration based on composition chart"""
    
    # Duration distribution (percentages)
    duration_splits = {
        DurationCategory.SHORT: 30,   # 30%: 10-60 seconds
        DurationCategory.MEDIUM: 50,  # 50%: 1-5 minutes  
        DurationCategory.LONG: 20     # 20%: 5+ minutes
    }
    
    # Duration ranges (seconds)
    duration_ranges = {
        DurationCategory.SHORT: (10, 60),
        DurationCategory.MEDIUM: (60, 300),  # 1-5 minutes
        DurationCategory.LONG: (300, 600)   # 5-10 minutes
    }
    
    # Speaker count distribution (percentages)
    speaker_splits = {
        1: 10,    # 10%: monologue
        2: 25,    # 25%: dialogue (part of 2-3 speakers = 50%)
        3: 25,    # 25%: trialogue (part of 2-3 speakers = 50%)
        4: 20,    # 20%: 4 speakers (part of 4+ speakers = 40%)
        5: 20     # 20%: 5 speakers (part of 4+ speakers = 40%)
    }
    
    # Noise level distribution (percentages)
    noise_splits = {
        NoiseLevel.CLEAN: 25,   # 30%: no noise
        NoiseLevel.LOW: 30,     # 30%: SNR 15-20 dB
        NoiseLevel.MEDIUM: 30,  # 30%: SNR 5-15 dB
        NoiseLevel.HIGH: 15     # 20%: SNR 0-5 dB
    }
    
    # SNR ranges for noise levels (dB)
    snr_ranges = {
        NoiseLevel.LOW: (15, 20),
        NoiseLevel.MEDIUM: (5, 15),
        NoiseLevel.HIGH: (0, 5)
    }
    
    # Language composition distribution (percentages)
    language_splits = {
        LanguageComposition.HINDI_ONLY: 15,      # 15%
        LanguageComposition.PUNJABI_ONLY: 15,    # 15% 
        LanguageComposition.ENGLISH_ONLY: 10,    # 10%
        LanguageComposition.BILINGUAL: 30,       # 30%
        LanguageComposition.TRILINGUAL: 30       # 30%
    }
    
    # Overlap configuration
    overlap_probability = 0.15  # 15% chance of overlaps in multi-speaker files
    overlap_duration_range = (0.5, 2.0)  # 0.5-2.0 seconds
    
    # Audio parameters
    sample_rate = 16000
    silence_duration_range = (0.3, 1.5)

class SpeakerClipManager:
    """Manages speaker audio clips to prevent bias from repeating same clips"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.speaker_clips = {}
        self.used_clips = {}  # Track used clips per speaker
        self._initialize_speaker_clips()
    
    def _initialize_speaker_clips(self):
        """Initialize clips for each speaker"""
        for speaker_id in self.df['speaker_id'].unique():
            speaker_df = self.df[self.df['speaker_id'] == speaker_id]
            self.speaker_clips[speaker_id] = speaker_df.index.tolist()
            self.used_clips[speaker_id] = set()
    
    def get_fresh_clip(self, speaker_id: str, language_filter: Optional[str] = None) -> Optional[pd.Series]:
        """Get a fresh (unused) clip for a speaker, optionally filtered by language"""
        available_clips = [idx for idx in self.speaker_clips[speaker_id] 
                          if idx not in self.used_clips[speaker_id]]
        
        if language_filter:
            # Filter by language (handle NaN values)
            available_clips = [idx for idx in available_clips 
                             if pd.notna(self.df.loc[idx, 'lang']) and self.df.loc[idx, 'lang'] == language_filter]
        
        if not available_clips:
            # Reset used clips if we've exhausted all clips for this speaker
            self.used_clips[speaker_id] = set()
            available_clips = [idx for idx in self.speaker_clips[speaker_id]]
            
            if language_filter:
                available_clips = [idx for idx in available_clips 
                                 if pd.notna(self.df.loc[idx, 'lang']) and self.df.loc[idx, 'lang'] == language_filter]
        
        if not available_clips:
            return None
            
        # Select random fresh clip
        selected_idx = random.choice(available_clips)
        self.used_clips[speaker_id].add(selected_idx)
        
        return self.df.loc[selected_idx]

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_generation.log'),
            logging.StreamHandler()
        ]
    )

def load_multilingual_datasets(dataset_folder: str) -> pd.DataFrame:
    """Load all parquet files from the Datasets folder with language detection"""
    logging.info("Loading multilingual datasets...")
    all_data = []
    
    dataset_path = Path(dataset_folder)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
    
    parquet_files = list(dataset_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_folder}")
    
    # Group by language folders for reporting
    language_stats = {}
    
    for parquet_file in parquet_files:
        logging.info(f"Loading {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        all_data.append(df)
        
        # Track language statistics
        parent_folder = parquet_file.parent.name
        if parent_folder not in language_stats:
            language_stats[parent_folder] = 0
        language_stats[parent_folder] += len(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Log statistics
    logging.info(f"Dataset Statistics:")
    logging.info(f"  Total parquet files: {len(parquet_files)}")
    logging.info(f"  Total records loaded: {len(combined_df)}")
    logging.info(f"  Languages/folders: {list(language_stats.keys())}")
    for lang, count in language_stats.items():
        logging.info(f"    - {lang}: {count} records")
    
    # Validate required columns
    required_columns = ['audio_filepath', 'speaker_id', 'lang']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logging.info(f"  Unique speakers: {combined_df['speaker_id'].nunique()}")
    logging.info(f"  Languages in data: {combined_df['lang'].unique().tolist()}")
    
    return combined_df

def extract_audio_from_bytes(audio_bytes) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract audio data from byte format"""
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes['bytes']))
        return audio_data, sr
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None, None

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio
    
    if LIBROSA_AVAILABLE:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    else:
        # Fallback to scipy
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples)

def generate_silence(duration: float, sample_rate: int) -> np.ndarray:
    """Generate silence with minimal background noise"""
    num_samples = int(duration * sample_rate)
    # Very low amplitude background noise to simulate silence
    silence = np.random.normal(0, 0.001, num_samples)
    return silence

def add_noise_to_audio(audio: np.ndarray, noise_level: NoiseLevel, config: DatasetConfiguration,
                      noise_dir: Optional[str] = None) -> np.ndarray:
    """Add noise to audio based on specified noise level and SNR"""
    if noise_level == NoiseLevel.CLEAN:
        return audio
    
    # Get SNR range for this noise level
    snr_range = config.snr_ranges[noise_level]
    target_snr = random.uniform(snr_range[0], snr_range[1])
    
    if AUDIOMENTATIONS_AVAILABLE and noise_dir and os.path.exists(noise_dir):
        # Use audiomentations with actual noise files
        try:
            augmentation = Compose([
                AddBackgroundNoise(
                    sounds_path=noise_dir,
                    min_snr_in_db=target_snr,
                    max_snr_in_db=target_snr + 2,
                    p=1.0
                )
            ])
            return augmentation(samples=audio, sample_rate=config.sample_rate)
        except Exception as e:
            logging.warning(f"Failed to add background noise: {e}. Using Gaussian noise.")
    
    # Fallback: Add Gaussian noise to achieve target SNR
    signal_power = np.mean(audio ** 2)
    signal_power_db = 10 * np.log10(signal_power + 1e-10)
    noise_power_db = signal_power_db - target_snr
    noise_power = 10 ** (noise_power_db / 10)
    
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Different noise types based on level
    if noise_level == NoiseLevel.LOW:
        # Mild background hum
        noise = noise * 0.5
    elif noise_level == NoiseLevel.MEDIUM:
        # Mix of noise types
        noise = noise + np.random.normal(0, np.sqrt(noise_power) * 0.3, len(audio))
    elif noise_level == NoiseLevel.HIGH:
        # More aggressive noise
        noise = noise + np.random.normal(0, np.sqrt(noise_power) * 0.5, len(audio))
    
    return audio + noise

def select_duration_category(config: DatasetConfiguration) -> DurationCategory:
    """Select duration category based on configured splits"""
    rand = random.random() * 100
    cumulative = 0
    
    for category, percentage in config.duration_splits.items():
        cumulative += percentage
        if rand <= cumulative:
            return category
    
    return DurationCategory.MEDIUM  # fallback

def select_speaker_count(config: DatasetConfiguration) -> int:
    """Select number of speakers based on configured splits"""
    rand = random.random() * 100
    cumulative = 0
    
    for count, percentage in config.speaker_splits.items():
        cumulative += percentage
        if rand <= cumulative:
            return count
    
    return 2  # fallback

def select_noise_level(config: DatasetConfiguration) -> NoiseLevel:
    """Select noise level based on configured splits"""
    rand = random.random() * 100
    cumulative = 0
    
    for level, percentage in config.noise_splits.items():
        cumulative += percentage
        if rand <= cumulative:
            return level
    
    return NoiseLevel.LOW  # fallback

def select_language_composition(config: DatasetConfiguration) -> LanguageComposition:
    """Select language composition based on configured splits"""
    rand = random.random() * 100
    cumulative = 0
    
    for composition, percentage in config.language_splits.items():
        cumulative += percentage
        if rand <= cumulative:
            return composition
    
    return LanguageComposition.BILINGUAL  # fallback

def get_target_duration(duration_category: DurationCategory, config: DatasetConfiguration) -> float:
    """Get target duration for the selected category"""
    min_dur, max_dur = config.duration_ranges[duration_category]
    return random.uniform(min_dur, max_dur)

def get_languages_for_composition(composition: LanguageComposition, available_langs: List[str]) -> List[str]:
    """Get list of languages for the specified composition"""
    # Handle empty language list
    if not available_langs:
        return ['unknown']
    
    # Normalize available languages
    available_langs = [lang.lower() for lang in available_langs if isinstance(lang, str)]
    
    if composition == LanguageComposition.HINDI_ONLY:
        return ['hindi', 'hi'] if any(lang in ['hindi', 'hi'] for lang in available_langs) else ([available_langs[0]] if available_langs else ['unknown'])
    elif composition == LanguageComposition.PUNJABI_ONLY:
        return ['punjabi', 'pa'] if any(lang in ['punjabi', 'pa'] for lang in available_langs) else ([available_langs[0]] if available_langs else ['unknown'])
    elif composition == LanguageComposition.ENGLISH_ONLY:
        return ['english', 'en'] if any(lang in ['english', 'en'] for lang in available_langs) else ([available_langs[0]] if available_langs else ['unknown'])
    elif composition == LanguageComposition.BILINGUAL:
        # Mix of 2 languages - prefer Hindi-English or Punjabi-English
        if 'hindi' in available_langs or 'hi' in available_langs:
            hindi = 'hindi' if 'hindi' in available_langs else 'hi'
            if 'english' in available_langs or 'en' in available_langs:
                english = 'english' if 'english' in available_langs else 'en'
                return [hindi, english]
        elif 'punjabi' in available_langs or 'pa' in available_langs:
            punjabi = 'punjabi' if 'punjabi' in available_langs else 'pa'
            if 'english' in available_langs or 'en' in available_langs:
                english = 'english' if 'english' in available_langs else 'en'
                return [punjabi, english]
        # Fallback to any 2 languages
        return available_langs[:2] if len(available_langs) >= 2 else available_langs
    elif composition == LanguageComposition.TRILINGUAL:
        # Mix of 3 languages - Hindi, Punjabi, English if available
        target_langs = []
        for lang_set in [['hindi', 'hi'], ['punjabi', 'pa'], ['english', 'en']]:
            for lang in lang_set:
                if lang in available_langs:
                    target_langs.append(lang)
                    break
        
        # Fill up to 3 languages
        while len(target_langs) < 3 and len(target_langs) < len(available_langs):
            for lang in available_langs:
                if lang not in target_langs:
                    target_langs.append(lang)
                    break
        
        return target_langs[:3] if target_langs else available_langs
    
    return available_langs

def create_mixed_language_audio(clip_manager: SpeakerClipManager, speakers: List[str], 
                              target_languages: List[str], target_duration: float,
                              config: DatasetConfiguration) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Create audio with mixed languages (code-switching)"""
    
    audio_segments = []
    diarization_records = []
    current_time = 0.0
    previous_speaker = None
    
    # Calculate total clips needed based on target duration
    avg_clip_duration = 3.0  # Assume average clip is 3 seconds
    total_clips_needed = int(target_duration / avg_clip_duration * 1.2)  # Add buffer
    
    clips_generated = 0
    
    # Create speaker-language preferences for more realistic diarization
    speaker_lang_preferences = {}
    if len(speakers) >= len(target_languages) and len(target_languages) > 1:
        # Assign each language to a preferred speaker for more realistic scenarios
        shuffled_speakers = speakers.copy()
        random.shuffle(shuffled_speakers)
        for i, lang in enumerate(target_languages):
            speaker_lang_preferences[lang] = shuffled_speakers[i % len(speakers)]
    
    while current_time < target_duration and clips_generated < total_clips_needed:
        # Select language for this segment (supports code-switching)
        target_lang = random.choice(target_languages)
        
        # Select speaker based on language preference if available
        if target_lang in speaker_lang_preferences:
            # Use preferred speaker for this language, but allow some variation
            if random.random() < 0.8:  # 80% chance to use preferred speaker
                speaker = speaker_lang_preferences[target_lang]
            else:
                # 20% chance for code-switching (same speaker, different language)
                available_speakers = [s for s in speakers if s != previous_speaker]
                if not available_speakers:
                    available_speakers = speakers
                speaker = random.choice(available_speakers)
        else:
            # Fallback: select speaker (prefer different from previous)
            available_speakers = [s for s in speakers if s != previous_speaker]
            if not available_speakers:
                available_speakers = speakers
            speaker = random.choice(available_speakers)
        
        # Get a fresh clip for this speaker in target language
        clip_row = clip_manager.get_fresh_clip(speaker, target_lang)
        if clip_row is None:
            # Try any language if specific language not available
            clip_row = clip_manager.get_fresh_clip(speaker)
            if clip_row is None:
                break
        
        # Extract and process audio
        audio_data, sr = extract_audio_from_bytes(clip_row['audio_filepath'])
        if audio_data is None:
            continue
            
        # Resample if needed
        if sr != config.sample_rate:
            audio_data = resample_audio(audio_data, sr, config.sample_rate)
        
        clip_duration = len(audio_data) / config.sample_rate
        
        # Decide if this clip should overlap with previous
        should_overlap = (
            len(audio_segments) > 0 and 
            random.random() < config.overlap_probability and
            previous_speaker != speaker and
            len(speakers) > 1  # Only overlap in multi-speaker scenarios
        )
        
        if should_overlap:
            # Create overlapping speech
            overlap_duration = random.uniform(*config.overlap_duration_range)
            overlap_duration = min(overlap_duration, clip_duration * 0.5)  # Max 50% overlap
            
            start_time = current_time - overlap_duration
            end_time = start_time + clip_duration
        else:
            # Normal sequential speech with silence gap
            if len(audio_segments) > 0:
                silence_duration = random.uniform(*config.silence_duration_range)
                current_time += silence_duration
            
            start_time = current_time
            end_time = current_time + clip_duration
        
        # Add segment
        audio_segments.append({
            'audio': audio_data,
            'start': start_time,
            'end': end_time,
            'speaker': speaker,
            'language': clip_row['lang']
        })
        
        current_time = end_time
        previous_speaker = speaker
        clips_generated += 1
    
    # Build final audio by mixing overlapping segments
    if not audio_segments:
        return np.array([]), [], speakers
    
    total_duration = max(seg['end'] for seg in audio_segments)
    total_samples = int(total_duration * config.sample_rate)
    final_audio = np.zeros(total_samples)
    
    # Mix all segments at their positions
    for seg in audio_segments:
        start_sample = int(seg['start'] * config.sample_rate)
        audio_data = seg['audio']
        end_sample = start_sample + len(audio_data)
        
        # Handle bounds
        if start_sample >= len(final_audio):
            continue
        if end_sample > len(final_audio):
            audio_data = audio_data[:len(final_audio) - start_sample]
            end_sample = len(final_audio)
        if len(audio_data) == 0:
            continue
        
        # Mix audio (overlaps will naturally blend)
        final_audio[start_sample:end_sample] += audio_data * 0.7
        
        # Record diarization info
        diarization_records.append({
            'Speaker': seg['speaker'],  # This will be mapped later in save_audio_and_annotations
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
    
    return final_audio, diarization_records, speakers

def save_audio_and_annotations(audio_data: np.ndarray, diarization_records: List[Dict], 
                              filename: str, output_dir: str, config: DatasetConfiguration,
                              speaker_mapping: Dict[str, str] = None) -> Tuple[str, str, str]:
    """Save audio file and generate both CSV and RTTM annotations"""
    
    # Create directory structure
    audio_dir = Path(output_dir) / "audio"
    csv_dir = Path(output_dir) / "csv" 
    rttm_dir = Path(output_dir) / "rttm"
    
    for dir_path in [audio_dir, csv_dir, rttm_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save audio file
    audio_filename = f"{filename}.wav"
    audio_path = audio_dir / audio_filename
    sf.write(str(audio_path), audio_data, config.sample_rate)
    
    # Save CSV file
    csv_data = []
    for idx, record in enumerate(diarization_records, start=1):
        # Map speaker ID if mapping is provided
        mapped_speaker = speaker_mapping.get(record['Speaker'], record['Speaker']) if speaker_mapping else record['Speaker']
        csv_data.append({
            'AudioFileName': audio_filename,
            'Speaker': mapped_speaker,
            'StartTS': f"{record['StartTS']:.3f}",
            'EndTS': f"{record['EndTS']:.3f}",
            'Language': record['Language']
        })
    
    csv_path = csv_dir / f"{filename}.csv"
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(str(csv_path), index=True, index_label='')
    
    # Save RTTM file
    rttm_path = rttm_dir / f"{filename}.rttm"
    with open(str(rttm_path), 'w') as f:
        for record in diarization_records:
            start = record['StartTS']
            end = record['EndTS']
            duration = end - start
            # Map speaker ID if mapping is provided
            mapped_speaker = speaker_mapping.get(record['Speaker'], record['Speaker']) if speaker_mapping else record['Speaker']
            f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <NA> <NA> {mapped_speaker} <NA> <NA>\n")
    
    logging.info(f"Saved: {audio_path}, {csv_path}, {rttm_path}")
    return str(audio_path), str(csv_path), str(rttm_path)

def append_to_combined_files(diarization_records: List[Dict], filename: str, output_dir: str, 
                           speaker_mapping: Dict[str, str] = None):
    """Append to combined CSV and RTTM files"""
    
    # Combined CSV
    combined_csv_path = Path(output_dir) / "all_samples_combined.csv"
    audio_filename = f"{filename}.wav"
    
    csv_data = []
    for record in diarization_records:
        # Map speaker ID if mapping is provided
        mapped_speaker = speaker_mapping.get(record['Speaker'], record['Speaker']) if speaker_mapping else record['Speaker']
        csv_data.append({
            'AudioFileName': audio_filename,
            'Speaker': mapped_speaker,
            'StartTS': f"{record['StartTS']:.3f}",
            'EndTS': f"{record['EndTS']:.3f}",
            'Language': record['Language']
        })
    
    df_new = pd.DataFrame(csv_data)
    
    # Append or create CSV
    if combined_csv_path.exists():
        df_existing = pd.read_csv(str(combined_csv_path), index_col=0)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(str(combined_csv_path), index=True, index_label='')
    else:
        df_new.to_csv(str(combined_csv_path), index=True, index_label='')
    
    # Combined RTTM
    combined_rttm_path = Path(output_dir) / "all_samples_combined.rttm"
    with open(str(combined_rttm_path), 'a') as f:
        for record in diarization_records:
            start = record['StartTS']
            end = record['EndTS']
            duration = end - start
            # Map speaker ID if mapping is provided
            mapped_speaker = speaker_mapping.get(record['Speaker'], record['Speaker']) if speaker_mapping else record['Speaker']
            f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <NA> <NA> {mapped_speaker} <NA> <NA>\n")

def create_speaker_id_mapping(speakers: List[str], use_simple_ids: bool = True) -> Dict[str, str]:
    """Create a mapping from original speaker IDs to simplified IDs"""
    # Filter out NaN values and ensure all are strings
    clean_speakers = [str(speaker) for speaker in speakers if pd.notna(speaker)]
    
    if not use_simple_ids:
        return {speaker: speaker for speaker in clean_speakers}
    
    speaker_mapping = {}
    for i, speaker in enumerate(sorted(set(clean_speakers))):
        speaker_mapping[speaker] = f"Speaker_{i:02d}"
    
    return speaker_mapping

def calculate_sample_counts(total_hours: float, config: DatasetConfiguration) -> Dict:
    """Calculate number of samples needed for each category combination"""
    
    # Estimate average duration for each category
    avg_durations = {
        DurationCategory.SHORT: (config.duration_ranges[DurationCategory.SHORT][0] + 
                                config.duration_ranges[DurationCategory.SHORT][1]) / 2,
        DurationCategory.MEDIUM: (config.duration_ranges[DurationCategory.MEDIUM][0] + 
                                 config.duration_ranges[DurationCategory.MEDIUM][1]) / 2,
        DurationCategory.LONG: (config.duration_ranges[DurationCategory.LONG][0] + 
                               config.duration_ranges[DurationCategory.LONG][1]) / 2
    }
    
    total_seconds = total_hours * 3600
    
    # Calculate samples per duration category
    samples_per_category = {}
    for category, percentage in config.duration_splits.items():
        category_seconds = total_seconds * (percentage / 100)
        avg_duration = avg_durations[category]
        samples_needed = int(category_seconds / avg_duration)
        samples_per_category[category] = samples_needed
    
    total_samples = sum(samples_per_category.values())
    
    logging.info(f"Sample distribution for {total_hours} hours:")
    for category, count in samples_per_category.items():
        logging.info(f"  {category.value}: {count} samples")
    logging.info(f"  Total samples: {total_samples}")
    
    return samples_per_category

def generate_dataset(total_hours: float, output_dir: str, source_dirs: Dict[str, str], 
                    noise_dir: Optional[str] = None, use_simple_speaker_ids: bool = False):
    """
    Main dataset generation function
    
    Args:
        total_hours: Total duration of dataset to generate
        output_dir: Output directory path
        source_dirs: Dictionary mapping language names to dataset directories
        noise_dir: Optional directory containing noise files
    """
    
    setup_logging()
    config = DatasetConfiguration()
    
    # Clear combined files if they exist
    combined_csv_path = Path(output_dir) / "all_samples_combined.csv"
    combined_rttm_path = Path(output_dir) / "all_samples_combined.rttm"
    
    for path in [combined_csv_path, combined_rttm_path]:
        if path.exists():
            path.unlink()
    
    # Load datasets
    all_datasets = []
    for lang_name, dataset_path in source_dirs.items():
        try:
            df = load_multilingual_datasets(dataset_path)
            all_datasets.append(df)
            logging.info(f"Loaded {lang_name} dataset: {len(df)} samples")
        except Exception as e:
            logging.error(f"Failed to load {lang_name} dataset from {dataset_path}: {e}")
    
    if not all_datasets:
        raise ValueError("No datasets loaded successfully")
    
    # Combine all datasets
    combined_df = pd.concat(all_datasets, ignore_index=True)
    logging.info(f"Combined dataset: {len(combined_df)} samples, {combined_df['speaker_id'].nunique()} unique speakers")
    
    # Initialize clip manager to prevent bias
    clip_manager = SpeakerClipManager(combined_df)
    
    # Create speaker ID mapping
    all_speakers = combined_df['speaker_id'].unique().tolist()
    speaker_id_mapping = create_speaker_id_mapping(all_speakers, use_simple_speaker_ids)
    logging.info(f"Created speaker ID mapping for {len(speaker_id_mapping)} speakers")
    
    # Calculate samples needed per category
    samples_per_duration = calculate_sample_counts(total_hours, config)
    
    # Track generation statistics
    stats = {
        'generated_samples': 0,
        'duration_categories': {cat: 0 for cat in DurationCategory},
        'speaker_counts': {count: 0 for count in config.speaker_splits.keys()},
        'noise_levels': {level: 0 for level in NoiseLevel},
        'language_compositions': {comp: 0 for comp in LanguageComposition},
        'total_duration': 0.0,
        'overlaps_created': 0
    }
    
    sample_counter = 1
    
    # Generate samples for each duration category
    for duration_category, num_samples in samples_per_duration.items():
        logging.info(f"Generating {num_samples} samples for {duration_category.value} category...")
        
        for i in range(num_samples):
            try:
                # Select parameters for this sample
                target_duration = get_target_duration(duration_category, config)
                speaker_count = select_speaker_count(config)
                noise_level = select_noise_level(config)
                language_composition = select_language_composition(config)
                
                # Get available languages, filter out NaN values
                available_langs = [lang for lang in combined_df['lang'].unique().tolist() if pd.notna(lang)]
                target_languages = get_languages_for_composition(language_composition, available_langs)
                
                # Ensure we have enough speakers for multilingual compositions
                min_speakers_needed = len(target_languages) if len(target_languages) > 1 else 1
                if speaker_count < min_speakers_needed:
                    speaker_count = min_speakers_needed
                    logging.info(f"Adjusted speaker count to {speaker_count} for {language_composition.value} composition")
                
                # Select speakers
                all_speakers = combined_df['speaker_id'].unique().tolist()
                if len(all_speakers) < speaker_count:
                    speaker_count = len(all_speakers)
                selected_speakers = random.sample(all_speakers, speaker_count)
                
                # Create mixed language audio
                audio_data, diarization_records, used_speakers = create_mixed_language_audio(
                    clip_manager, selected_speakers, target_languages, target_duration, config
                )
                
                # Create per-file speaker mapping if using simple IDs
                if use_simple_speaker_ids:
                    file_speakers = list(set(record['Speaker'] for record in diarization_records))
                    file_speaker_mapping = {speaker: f"Speaker_{i:02d}" for i, speaker in enumerate(sorted(file_speakers))}
                else:
                    file_speaker_mapping = speaker_id_mapping
                
                if len(audio_data) == 0:
                    logging.warning(f"Failed to generate audio for sample {sample_counter}")
                    continue
                
                # Add noise
                audio_data = add_noise_to_audio(audio_data, noise_level, config, noise_dir)
                
                # Save files
                filename = f"diarization_sample_{sample_counter:04d}"
                save_audio_and_annotations(audio_data, diarization_records, filename, output_dir, config, file_speaker_mapping)
                append_to_combined_files(diarization_records, filename, output_dir, file_speaker_mapping)
                
                # Update statistics
                actual_duration = len(audio_data) / config.sample_rate
                overlap_count = sum(1 for i, r1 in enumerate(diarization_records) 
                                  for r2 in diarization_records[i+1:] 
                                  if r1['StartTS'] < r2['EndTS'] and r2['StartTS'] < r1['EndTS'] 
                                  and r1['Speaker'] != r2['Speaker'])
                
                stats['generated_samples'] += 1
                stats['duration_categories'][duration_category] += 1
                stats['speaker_counts'][len(used_speakers)] += 1
                stats['noise_levels'][noise_level] += 1
                stats['language_compositions'][language_composition] += 1
                stats['total_duration'] += actual_duration
                stats['overlaps_created'] += overlap_count
                
                logging.info(f"Sample {sample_counter}: {duration_category.value}, "
                           f"{len(used_speakers)} speakers, {noise_level.value} noise, "
                           f"{language_composition.value}, {actual_duration:.1f}s, {overlap_count} overlaps")
                
                sample_counter += 1
                
            except Exception as e:
                logging.error(f"Failed to generate sample {sample_counter}: {e}")
                continue
    
    # Log final statistics
    logging.info("\n" + "="*50)
    logging.info("DATASET GENERATION COMPLETE")
    logging.info("="*50)
    logging.info(f"Total samples generated: {stats['generated_samples']}")
    logging.info(f"Total duration: {stats['total_duration']/3600:.2f} hours")
    logging.info(f"Total overlaps created: {stats['overlaps_created']}")
    
    logging.info("\nDuration Category Distribution:")
    for cat, count in stats['duration_categories'].items():
        percentage = (count / stats['generated_samples'] * 100) if stats['generated_samples'] > 0 else 0
        logging.info(f"  {cat.value}: {count} samples ({percentage:.1f}%)")
    
    logging.info("\nSpeaker Count Distribution:")
    for count, num_samples in stats['speaker_counts'].items():
        percentage = (num_samples / stats['generated_samples'] * 100) if stats['generated_samples'] > 0 else 0
        logging.info(f"  {count} speakers: {num_samples} samples ({percentage:.1f}%)")
    
    logging.info("\nNoise Level Distribution:")
    for level, count in stats['noise_levels'].items():
        percentage = (count / stats['generated_samples'] * 100) if stats['generated_samples'] > 0 else 0
        logging.info(f"  {level.value}: {count} samples ({percentage:.1f}%)")
    
    logging.info("\nLanguage Composition Distribution:")
    for comp, count in stats['language_compositions'].items():
        percentage = (count / stats['generated_samples'] * 100) if stats['generated_samples'] > 0 else 0
        logging.info(f"  {comp.value}: {count} samples ({percentage:.1f}%)")
    
    logging.info(f"\nOutput saved to: {output_dir}")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Generate multilingual speaker diarization dataset")
    
    parser.add_argument('--total_hours', type=float, default=1.0,
                       help='Total hours of dataset to generate (default: 1.0)')
    parser.add_argument('--output_dir', type=str, default='output_diarization_dataset',
                       help='Output directory (default: output_diarization_dataset)')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets',
                       help='Path to datasets directory (default: ../Datasets)')
    parser.add_argument('--noise_dir', type=str, default=None,
                       help='Path to noise files directory (optional)')
    parser.add_argument('--simple_speaker_ids', action='store_true',
                       help='Use simple Speaker_00, Speaker_01 format instead of original IDs')
    
    args = parser.parse_args()
    
    # Setup source directories (modify as needed for your dataset structure)
    source_dirs = {
        'multilingual': args.dataset_dir
    }
    
    # Generate dataset
    generate_dataset(
        total_hours=args.total_hours,
        output_dir=args.output_dir,
        source_dirs=source_dirs,
        noise_dir=args.noise_dir,
        use_simple_speaker_ids=args.simple_speaker_ids
    )

if __name__ == "__main__":
    main()