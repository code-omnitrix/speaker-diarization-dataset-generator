#!/usr/bin/env python3
"""
Quick diagnostic script to check the status of the dataset generation process
"""

import os
import psutil
import time
from pathlib import Path

def check_process_status():
    """Check the status of running Python processes"""
    print("🔍 Checking running Python processes...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                runtime = time.time() - proc.info['create_time']
                
                print(f"📊 Process ID: {proc.info['pid']}")
                print(f"   Memory: {memory_mb:.1f} MB")
                print(f"   CPU: {proc.info['cpu_percent']:.1f}%")
                print(f"   Runtime: {runtime/60:.1f} minutes")
                
                # Check if this is our dataset generation process
                if memory_mb > 1000:  # More than 1GB memory usage
                    print(f"   🚨 Large process detected - likely dataset generation")
                    
                    # Check CPU usage over time
                    cpu_samples = []
                    for i in range(5):
                        cpu_samples.append(psutil.Process(proc.info['pid']).cpu_percent(interval=1))
                    
                    avg_cpu = sum(cpu_samples) / len(cpu_samples)
                    print(f"   📈 Average CPU over 5 seconds: {avg_cpu:.1f}%")
                    
                    if avg_cpu < 1.0:
                        print("   ⚠️  Process appears to be stuck (very low CPU usage)")
                        print("   💡 Consider terminating and restarting with smaller batch size")
                    else:
                        print("   ✅ Process appears to be actively working")
                
                print("   " + "-" * 50)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def check_output_files():
    """Check the current state of output files"""
    print("\n📁 Checking output files...")
    
    output_dir = Path("output_diarization_dataset")
    
    if not output_dir.exists():
        print("❌ Output directory doesn't exist yet")
        return
    
    # Check directories
    for subdir in ['audio', 'csv', 'rttm']:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            print(f"📂 {subdir}/: {len(files)} files")
            
            # Show latest files
            if files:
                latest_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
                for f in latest_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    mtime = time.ctime(f.stat().st_mtime)
                    print(f"   📄 {f.name} ({size_mb:.2f}MB) - {mtime}")
        else:
            print(f"❌ {subdir}/ directory doesn't exist")
    
    # Check for combined files
    combined_files = ['all_samples_combined.csv', 'all_samples_combined.rttm']
    for filename in combined_files:
        file_path = output_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            mtime = time.ctime(file_path.stat().st_mtime)
            print(f"📄 {filename} ({size_mb:.2f}MB) - {mtime}")

def check_log_file():
    """Check the current log file status"""
    print("\n📋 Checking log file...")
    
    log_file = Path("dataset_generation.log")
    if log_file.exists():
        size_mb = log_file.stat().st_size / (1024 * 1024)
        mtime = time.ctime(log_file.stat().st_mtime)
        print(f"📄 {log_file.name} ({size_mb:.2f}MB) - Last modified: {mtime}")
        
        # Show last few lines
        print("\n📖 Last 10 lines of log:")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("❌ Log file not found")

if __name__ == "__main__":
    print("🚀 Dataset Generation Process Diagnostics")
    print("=" * 50)
    
    check_process_status()
    check_output_files()
    check_log_file()
    
    print("\n💡 Recommendations:")
    print("1. If process is stuck (low CPU), terminate and restart")
    print("2. If memory usage is very high, consider reducing batch size")
    print("3. Check log file for any error messages")
    print("4. Ensure sufficient disk space for output files")