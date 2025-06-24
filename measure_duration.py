#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import audio

def main():
    # Directory to analyze
    recordings_dir = "../recordings_2025-06-19"
    
    # Check if directory exists
    if not os.path.exists(recordings_dir):
        print(f"Error: Directory '{recordings_dir}' does not exist")
        sys.exit(1)
    
    # Find all audio files (common extensions)
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.mp4'}
    audio_files = []
    
    for root, dirs, files in os.walk(recordings_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in '{recordings_dir}'")
        return
    
    print(f"Found {len(audio_files)} audio files in '{recordings_dir}'")
    print("Analyzing durations...")
    
    total_duration = 0
    processed_files = 0
    failed_files = 0
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"Processing {i}/{len(audio_files)}: {os.path.basename(file_path)}", end="... ")
        
        duration = audio.get_duration(file_path)
        if duration is not None:
            total_duration += duration
            processed_files += 1
            print(f"{duration:.2f}s")
        else:
            failed_files += 1
            print("FAILED")
    
    # Convert to minutes and hours
    total_minutes = total_duration / 60
    total_hours = total_minutes / 60
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Total files found: {len(audio_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed to process: {failed_files}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total duration: {total_minutes:.2f} minutes")
    print(f"Total duration: {total_hours:.2f} hours")
    print("="*50)

if __name__ == "__main__":
    main() 