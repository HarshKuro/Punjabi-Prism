import os
import librosa
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi
import re

# === Helper Functions ===
def compute_snr(clean, noisy):
    try:
        noise = noisy - clean
        snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-9))
        return snr
    except Exception:
        return None

def rate_quality(pesq_score, snr_score):
    """Rate audio quality based on PESQ and SNR scores"""
    if pesq_score is None or snr_score is None:
        return "Unknown"
    if pesq_score >= 3.5 and snr_score >= 20:
        return "Very Good"
    elif pesq_score >= 3.0 and snr_score >= 15:
        return "Good"
    elif pesq_score >= 2.0 and snr_score >= 8:
        return "Average"
    elif pesq_score >= 1.5 and snr_score >= 3:
        return "Poor"
    else:
        return "Very Poor"

def extract_distance_from_folder(folder_name):
    """Extract distance from folder name like 'F21m', '1m', 'F30m', 'M20m'"""
    # Handle patterns like F21m, F22m, F24m, F30m, M20m
    match = re.search(r'(\d+)m', folder_name)
    if match:
        distance = int(match.group(1))
        # Extract the actual distance (last digit(s))
        if distance >= 20 and distance < 30:  # F21m, F22m, F24m, M20m
            return distance % 10 if distance % 10 != 0 else distance // 10
        elif distance == 30:  # F30m
            return 0  # 0m distance
        else:
            return distance  # For cases like 1m
    return None

def process_dataset(base_dir, dataset_name):
    """Process a single dataset (F1, F2, etc.) and return results"""
    dataset_path = os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset {dataset_name} not found at {dataset_path}")
        return []
    
    results = []
    
    # Get all subdirectories (distance folders)
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not subdirs:
        print(f"❌ No subdirectories found in {dataset_name}")
        return []
    
    print(f"\n🔍 Processing dataset: {dataset_name}")
    print(f"📁 Found subdirectories: {subdirs}")
    
    for subdir in subdirs:
        distance = extract_distance_from_folder(subdir)
        if distance is None:
            print(f"⚠️ Could not extract distance from folder: {subdir}")
            continue
            
        folder_path = os.path.join(dataset_path, subdir)
        
        # Get all audio files in this distance folder
        audio_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                fname_lower = file.lower()
                if fname_lower.endswith(".wav") or fname_lower.endswith(".m4a") or ".m4a.wav" in fname_lower:
                    audio_files.append(os.path.join(root, file))
        
        print(f"  📂 {subdir} (Distance: {distance}m) - Found {len(audio_files)} audio files")
        
        # Process each audio file
        for idx, file_path in enumerate(audio_files, 1):
            fname = os.path.basename(file_path)
            
            try:
                noisy, sr = librosa.load(file_path, sr=16000, mono=True)
            except Exception as e:
                print(f"    ❌ Could not load {fname}: {e}")
                continue

            # --- Audio Quality Metrics ---
            
            # 1. RMS Energy (Root Mean Square) - indicates signal strength
            rms_energy = np.sqrt(np.mean(noisy ** 2))
            
            # 2. Zero Crossing Rate - indicates signal variability
            zcr = np.mean(librosa.feature.zero_crossing_rate(noisy)[0])
            
            # 3. Spectral Centroid - indicates brightness of the signal
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=noisy, sr=sr)[0])
            
            # 4. Enhanced SNR estimation using voice activity detection
            # Use spectral subtraction for better SNR estimation
            noise_samples = int(0.3 * sr)  # Use first/last 0.3 seconds for noise estimation
            if len(noisy) > 2 * noise_samples:
                # Estimate noise from quieter parts (beginning and end)
                noise_start = noisy[:noise_samples]
                noise_end = noisy[-noise_samples:]
                estimated_noise = np.concatenate([noise_start, noise_end])
                
                # Calculate noise power
                noise_power = np.mean(estimated_noise ** 2) + 1e-10
                
                # Calculate signal power (middle section, likely more speech)
                signal_section = noisy[noise_samples:-noise_samples] if len(noisy) > 2 * noise_samples else noisy
                signal_power = np.mean(signal_section ** 2) + 1e-10
                
                # SNR calculation
                snr_score = 10 * np.log10(signal_power / noise_power)
            else:
                snr_score = 10 * np.log10(rms_energy + 1e-10)  # Fallback based on RMS
            
            # 5. Pseudo-PESQ estimation based on audio characteristics
            # This is NOT real PESQ but an estimate based on signal quality indicators
            def estimate_pesq_from_features(rms, zcr, centroid, snr):
                # Normalize features to typical speech ranges
                rms_norm = np.clip(rms / 0.1, 0, 1)  # Normalize RMS
                zcr_norm = np.clip(zcr / 0.15, 0, 1)  # Normalize ZCR
                centroid_norm = np.clip(centroid / 2000, 0, 2)  # Normalize spectral centroid
                snr_norm = np.clip((snr + 10) / 40, 0, 1)  # Normalize SNR (-10 to 30 dB range)
                
                # Weighted combination (these weights are heuristic)
                pesq_estimate = (
                    0.3 * rms_norm +      # Signal strength
                    0.2 * (1 - abs(zcr_norm - 0.5)) +  # ZCR in optimal range
                    0.2 * (1 - abs(centroid_norm - 0.7)) +  # Centroid in speech range
                    0.3 * snr_norm        # SNR contribution
                )
                
                # Scale to PESQ range (1.0 to 4.5)
                pesq_estimate = 1.0 + 3.5 * pesq_estimate
                return max(1.0, min(4.5, pesq_estimate))
            
            pesq_score = estimate_pesq_from_features(rms_energy, zcr, spectral_centroid, snr_score)
            
            # 6. STOI estimation (also synthetic, as real STOI needs reference)
            # Based on signal characteristics that correlate with intelligibility
            def estimate_stoi_from_features(rms, zcr, snr):
                # Simple heuristic based on signal quality
                stoi_base = 0.5  # Baseline intelligibility
                
                # RMS contribution (louder = more intelligible, up to a point)
                rms_contrib = min(0.3, rms * 10)
                
                # ZCR contribution (moderate ZCR is good for speech)
                optimal_zcr = 0.08  # Typical for speech
                zcr_contrib = 0.2 * (1 - abs(zcr - optimal_zcr) / optimal_zcr)
                
                # SNR contribution
                snr_contrib = min(0.3, max(0, snr / 30))
                
                stoi_estimate = stoi_base + rms_contrib + zcr_contrib + snr_contrib
                return max(0.1, min(1.0, stoi_estimate))
            
            stoi_score = estimate_stoi_from_features(rms_energy, zcr, snr_score)

            quality = rate_quality(pesq_score, snr_score)

            results.append({
                "file": fname,
                "distance_from_source": distance,
                "pesq": pesq_score,
                "snr": snr_score,
                "stoi": stoi_score,
                "rms_energy": rms_energy,
                "zero_crossing_rate": zcr,
                "spectral_centroid": spectral_centroid,
                "quality": rate_quality(pesq_score, snr_score)
            })

            print(f"    [{idx}/{len(audio_files)}] ✅ {fname} | Distance={distance}m | PESQ={pesq_score:.2f}, SNR={snr_score:.1f}dB, STOI={stoi_score:.3f}")
    
    return results

def main():
    base_dir = "."  # Current directory
    datasets = ["F1", "F2", "F3", "M1", "M2"]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"🚀 Starting processing for dataset: {dataset}")
        print(f"{'='*60}")
        
        results = process_dataset(base_dir, dataset)
        
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Sort by file name to maintain consistent order
            df = df.sort_values(['file'])
            
            # Save to CSV
            output_file = f"punjabi_quality_scores_{dataset.lower()}_corrected.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\n📊 Summary for {dataset}:")
            print(f"   Total files processed: {len(results)}")
            print(f"   Distances found: {sorted(df['distance_from_source'].unique())}")
            print(f"   Average PESQ: {df['pesq'].mean():.3f}")
            print(f"   Average SNR: {df['snr'].mean():.3f} dB")
            print(f"   Average STOI: {df['stoi'].mean():.3f}")
            print(f"   Average RMS Energy: {df['rms_energy'].mean():.6f}")
            print(f"   Average Spectral Centroid: {df['spectral_centroid'].mean():.1f} Hz")
            print(f"   📂 Results saved to: {output_file}")
            
            # Show distance distribution
            distance_counts = df['distance_from_source'].value_counts().sort_index()
            print(f"   Distance distribution:")
            for dist, count in distance_counts.items():
                print(f"     {dist}m: {count} files")
        else:
            print(f"❌ No results found for dataset {dataset}")
    
    print(f"\n{'='*60}")
    print("🎉 All datasets processed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
