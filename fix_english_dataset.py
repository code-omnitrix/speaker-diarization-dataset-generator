#!/usr/bin/env python3
"""
🔧 Fix English Dataset Structure
===============================

This script fixes the English parquet files to match the expected structure
for the diarization dataset generation.

Issues to fix:
1. Missing 'speaker_id' column
2. Missing 'lang' column (but has 'primary_language')
3. Ensure compatibility with Hindi/Punjabi structure
"""

import pandas as pd
import os
from pathlib import Path

def fix_english_dataset():
    """Fix English parquet files to match expected structure"""
    
    english_dir = Path(r"C:\Users\saksh\OneDrive\Desktop\Projects\Audio Agnostic\data-preparation-finetuning-pyannote\Datasets\english")
    
    print("🔧 Fixing English dataset structure...")
    print(f"📂 English directory: {english_dir}")
    
    # Get all parquet files
    parquet_files = list(english_dir.glob("*.parquet"))
    print(f"📊 Found {len(parquet_files)} parquet files")
    
    for i, parquet_file in enumerate(parquet_files):
        print(f"\n🔄 Processing {parquet_file.name} ({i+1}/{len(parquet_files)})...")
        
        try:
            # Load the file
            df = pd.read_parquet(parquet_file)
            print(f"   📈 Original shape: {df.shape}")
            print(f"   📋 Original columns: {list(df.columns)}")
            
            # Check if already fixed
            if 'speaker_id' in df.columns and 'lang' in df.columns:
                print(f"   ✅ Already fixed - skipping")
                continue
            
            # Add missing speaker_id column
            if 'speaker_id' not in df.columns:
                # Generate speaker IDs based on row index (simple approach)
                # In real scenario, you might want more sophisticated speaker assignment
                df['speaker_id'] = df.index.astype(str)
                print(f"   ➕ Added speaker_id column")
            
            # Add missing lang column
            if 'lang' not in df.columns:
                if 'primary_language' in df.columns:
                    # Map primary_language to lang with proper code
                    df['lang'] = df['primary_language'].apply(lambda x: 'en' if 'english' in str(x).lower() else 'en')
                else:
                    # Default to English
                    df['lang'] = 'en'
                print(f"   ➕ Added lang column (set to 'en')")
            
            print(f"   📈 New shape: {df.shape}")
            print(f"   📋 New columns: {list(df.columns)}")
            
            # Save the fixed file
            backup_path = parquet_file.with_suffix('.parquet.backup')
            if not backup_path.exists():
                # Create backup
                df_original = pd.read_parquet(parquet_file)
                df_original.to_parquet(backup_path)
                print(f"   💾 Created backup: {backup_path.name}")
            
            # Save fixed version
            df.to_parquet(parquet_file)
            print(f"   ✅ Fixed and saved: {parquet_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {parquet_file.name}: {e}")
    
    print(f"\n🎉 English dataset fix completed!")
    print(f"\n📊 Verification - checking final structure:")
    
    # Verify the fix
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            has_speaker_id = 'speaker_id' in df.columns
            has_lang = 'lang' in df.columns
            lang_values = df['lang'].unique() if has_lang else []
            
            print(f"   📁 {parquet_file.name}:")
            print(f"      ✅ speaker_id: {has_speaker_id}")
            print(f"      ✅ lang: {has_lang}")
            if has_lang:
                print(f"      🌍 lang values: {list(lang_values)}")
        except Exception as e:
            print(f"   ❌ Error verifying {parquet_file.name}: {e}")

def main():
    """Main function"""
    print("🔧 English Dataset Structure Fix")
    print("=" * 40)
    
    # Ask for confirmation
    response = input("🤔 Do you want to fix the English dataset structure? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled.")
        return
    
    fix_english_dataset()
    
    print("\n" + "=" * 40)
    print("✅ Now you can regenerate your dataset with English support!")
    print("🚀 Run: python prepare_diarization_dataset_new.py --total_hours 15 --simple_speaker_ids")

if __name__ == "__main__":
    main()