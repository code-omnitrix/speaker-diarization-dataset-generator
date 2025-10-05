#!/usr/bin/env python3
"""
🧪 Test Hugging Face Upload - Quick Validation
==============================================

This script tests the basic functionality before running the full upload.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def test_dataset_structure():
    """Test if the dataset structure is correct"""
    print("🔍 Testing dataset structure...")
    
    dataset_path = Path("./output_diarization_dataset")
    
    # Check if path exists
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    # Check required files
    csv_file = dataset_path / "all_samples_combined.csv"
    audio_dir = dataset_path / "audio"
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return False
    
    # Load and check CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded successfully: {len(df)} rows")
        
        # Check columns
        required_cols = ['AudioFileName', 'Speaker', 'StartTS', 'EndTS', 'Language']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print(f"✅ All required columns present")
        
        # Check audio files
        unique_files = df['AudioFileName'].unique()
        audio_files = list(audio_dir.glob("*.wav"))
        
        print(f"✅ CSV references {len(unique_files)} audio files")
        print(f"✅ Found {len(audio_files)} actual audio files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

def test_packages():
    """Test if required packages are available"""
    print("📦 Testing required packages...")
    
    required_packages = [
        'huggingface_hub',
        'datasets', 
        'librosa',
        'soundfile',
        'pandas',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - Missing!")
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def test_huggingface_login():
    """Test Hugging Face authentication"""
    print("🔐 Testing Hugging Face authentication...")
    
    try:
        from huggingface_hub import HfApi
        
        # Try to create API instance
        api = HfApi()
        print("✅ HuggingFace Hub API initialized")
        
        # Test if already logged in
        try:
            whoami = api.whoami()
            print(f"✅ Already logged in as: {whoami['name']}")
            return True
        except Exception:
            print("ℹ️  Not logged in yet - you'll need to provide token during upload")
            return True
            
    except Exception as e:
        print(f"❌ HuggingFace Hub error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Pre-Upload Validation Tests")
    print("=" * 40)
    
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Required Packages", test_packages),
        ("HuggingFace Setup", test_huggingface_login)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        results[test_name] = test_func()
    
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 40)
    
    if all_passed:
        print("🎉 All tests passed! Ready to upload to Hugging Face!")
        print("\nNext steps:")
        print("1. Run: python upload_to_huggingface.py")
        print("2. Provide your HuggingFace token when prompted")
        print("3. Confirm upload settings")
    else:
        print("❌ Some tests failed. Please fix the issues above before uploading.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)