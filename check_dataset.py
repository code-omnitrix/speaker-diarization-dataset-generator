"""
Quick verification script to check if your Datasets folder is properly structured
"""

import sys
from pathlib import Path

def check_dataset_structure(dataset_folder="../Datasets"):
    """Check if the dataset folder is properly structured"""
    
    print("🔍 Checking dataset structure...\n")
    
    dataset_path = Path(dataset_folder)
    
    # Check if folder exists
    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_folder}")
        print(f"   Please create it or adjust the path in prepare_diarization_dataset.py")
        return False
    
    print(f"✅ Dataset folder exists: {dataset_path.absolute()}\n")
    
    # Find parquet files
    parquet_files = list(dataset_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {dataset_folder}")
        print(f"   Please add .parquet files to the folder")
        return False
    
    print(f"✅ Found {len(parquet_files)} parquet file(s):\n")
    
    # Show structure
    folders = {}
    for pf in parquet_files:
        parent = pf.parent.name if pf.parent.name != "Datasets" else "root"
        if parent not in folders:
            folders[parent] = []
        folders[parent].append(pf.name)
    
    for folder, files in folders.items():
        print(f"  📁 {folder}/")
        for file in files:
            print(f"     - {file}")
    
    print(f"\n✅ Dataset structure looks good!")
    print(f"\n💡 You can now run: python prepare_diarization_dataset.py")
    
    return True

if __name__ == "__main__":
    check_dataset_structure()
