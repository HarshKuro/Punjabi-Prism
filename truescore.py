"""
True PESQ and SNR Calculator
Based on ITU-T P.862 PESQ Standard and classical SNR formula.

Uses F3 (0m distance) files as clean references to calculate
proper PESQ and SNR scores for F1, F2, and M2 datasets.

Author: Based on ITU-T standards
"""

import os
import numpy as np
import pandas as pd
import librosa
from pesq import pesq
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_audio(file_path, target_sr=16000):
    """
    Load audio file and preprocess for PESQ calculation.
    PESQ requires mono audio at 16kHz sampling rate.
    """
    try:
        # Load audio with librosa (handles various formats)
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize audio to prevent clipping
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def calculate_true_snr(reference, degraded):
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB using ITU-T formula.
    
    SNR (dB) = 10 * log10(sum(reference^2) / sum((reference - degraded)^2))
    
    Parameters:
        reference (np.ndarray): Clean/reference signal
        degraded (np.ndarray): Noisy/degraded signal
        
    Returns:
        float: SNR value in decibels (dB)
    """
    # Ensure same length by taking minimum
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Calculate noise signal
    noise = reference - degraded
    
    # Avoid division by zero
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')  # Perfect signal
    
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value

def calculate_true_pesq(reference, degraded, sr):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) using ITU-T P.862.
    
    Parameters:
        reference (np.ndarray): Clean/reference signal
        degraded (np.ndarray): Noisy/degraded signal
        sr (int): Sampling rate (must be 8000 or 16000 Hz)
        
    Returns:
        float: PESQ MOS score (1.0 - 4.5)
    """
    try:
        # PESQ requires specific sampling rates
        if sr == 16000:
            mode = 'wb'  # Wideband
        elif sr == 8000:
            mode = 'nb'  # Narrowband
        else:
            # Resample to 16000 Hz for wideband PESQ
            reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
            degraded = librosa.resample(degraded, orig_sr=sr, target_sr=16000)
            sr = 16000
            mode = 'wb'
        
        # Calculate PESQ score
        score = pesq(sr, reference, degraded, mode)
        return score
    except Exception as e:
        print(f"PESQ calculation error: {e}")
        return None

def find_matching_reference(degraded_file, reference_dir):
    """
    Find matching F3 reference file for a degraded file.
    Matches based on speaker ID (S01, S02, etc.)
    """
    # Extract speaker ID from degraded filename
    match = re.search(r'pa_(S\d+)_', degraded_file)
    if not match:
        return None
    
    speaker_id = match.group(1)
    
    # Look for matching F3 file
    reference_pattern = f"pa_{speaker_id}_f3_female_*.wav"
    reference_files = list(Path(reference_dir).glob(reference_pattern))
    
    if reference_files:
        return str(reference_files[0])
    
    return None

def process_dataset_with_references(dataset_name, dataset_path, reference_path):
    """
    Process a dataset by calculating true PESQ and SNR using F3 references.
    """
    print(f"\n🔍 Processing {dataset_name} with F3 references...")
    
    results = []
    processed_count = 0
    
    # Get all subdirectories in the dataset
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        
        # Extract distance from folder name
        distance_match = re.search(r'(\d+)m', subdir)
        if distance_match:
            distance = int(distance_match.group(1))
        else:
            distance = 0
        
        print(f"  📂 {subdir} (Distance: {distance}m)")
        
        # Get all audio files in subdirectory
        audio_files = [f for f in os.listdir(subdir_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
        
        for audio_file in audio_files:
            audio_path = os.path.join(subdir_path, audio_file)
            
            # Find matching reference file
            reference_file = find_matching_reference(audio_file, reference_path)
            
            if not reference_file:
                print(f"    ❌ No reference found for {audio_file}")
                continue
            
            # Load degraded audio
            degraded_audio, sr_deg = load_and_preprocess_audio(audio_path)
            if degraded_audio is None:
                continue
            
            # Load reference audio
            reference_audio, sr_ref = load_and_preprocess_audio(reference_file)
            if reference_audio is None:
                continue
            
            # Calculate true PESQ and SNR
            try:
                pesq_score = calculate_true_pesq(reference_audio, degraded_audio, sr_ref)
                snr_score = calculate_true_snr(reference_audio, degraded_audio)
                
                if pesq_score is not None:
                    results.append({
                        'file': audio_file,
                        'distance_from_source': distance,
                        'pesq': pesq_score,
                        'snr': snr_score,
                        'reference_file': os.path.basename(reference_file)
                    })
                    
                    processed_count += 1
                    print(f"    ✅ [{processed_count}] {audio_file} | PESQ={pesq_score:.3f}, SNR={snr_score:.1f}dB")
                else:
                    print(f"    ❌ PESQ calculation failed for {audio_file}")
                    
            except Exception as e:
                print(f"    ❌ Error processing {audio_file}: {e}")
    
    return results

def main():
    """
    Main function to process all datasets with true PESQ/SNR calculations.
    """
    print("🚀 True PESQ and SNR Calculator")
    print("Using F3 (0m distance) as clean references")
    print("=" * 60)
    
    # Define paths
    base_path = os.getcwd()
    reference_path = os.path.join(base_path, "F3", "F30m")  # F3 0m distance files
    
    # Check if reference directory exists
    if not os.path.exists(reference_path):
        print(f"❌ Reference directory not found: {reference_path}")
        return
    
    print(f"📁 Reference directory: {reference_path}")
    reference_files = [f for f in os.listdir(reference_path) if f.endswith('.wav')]
    print(f"📊 Found {len(reference_files)} reference files in F3")
    
    # Define datasets to process
    datasets = {
        'F1': os.path.join(base_path, "F1"),
        'F2': os.path.join(base_path, "F2"),
        'M2': os.path.join(base_path, "M2")
    }
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found: {dataset_path}")
            continue
        
        # Process dataset with references
        results = process_dataset_with_references(dataset_name, dataset_path, reference_path)
        
        if results:
            # Create DataFrame and save results
            df = pd.DataFrame(results)
            output_file = f"true_scores_{dataset_name.lower()}.csv"
            df.to_csv(output_file, index=False)
            
            # Print summary statistics
            print(f"\n📊 Summary for {dataset_name}:")
            print(f"   Files processed: {len(results)}")
            print(f"   Average PESQ: {df['pesq'].mean():.3f}")
            print(f"   Average SNR: {df['snr'].mean():.1f} dB")
            print(f"   PESQ range: {df['pesq'].min():.3f} - {df['pesq'].max():.3f}")
            print(f"   SNR range: {df['snr'].min():.1f} - {df['snr'].max():.1f} dB")
            print(f"   📂 Results saved to: {output_file}")
            
            # Show distance-wise breakdown
            if 'distance_from_source' in df.columns:
                print(f"   Distance breakdown:")
                for dist in sorted(df['distance_from_source'].unique()):
                    dist_data = df[df['distance_from_source'] == dist]
                    print(f"     {dist}m: {len(dist_data)} files, "
                          f"PESQ={dist_data['pesq'].mean():.3f}, "
                          f"SNR={dist_data['snr'].mean():.1f}dB")
        else:
            print(f"❌ No results generated for {dataset_name}")
    
    print("\n🎉 True PESQ and SNR calculation completed!")

if __name__ == "__main__":
    main()
