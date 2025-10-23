import os
import librosa
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi

# === Paths ===
input_dir = "M20m"  # Updated to use the 1m folder with F1 data
results_csv = "punjabi_quality_scores_M2_0m.csv"

# === Helper Functions ===
def compute_snr(clean, noisy):
    try:
        noise = noisy - clean
        snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-9))
        return snr
    except Exception:
        return None

def rate_quality(pesq_score, snr_score):
    if pesq_score is None or snr_score is None:
        return "Unknown"
    if pesq_score >= 3.5 and snr_score >= 20:
        return "Very Good"
    elif pesq_score >= 3.0 and snr_score >= 15:
        return "Good"
    elif pesq_score >= 2.0 and snr_score >= 8:
        return "Average"
    elif pesq_score >= 1.5 and snr_score >= 3:
        return "Bad"
    else:
        return "Very Bad"

# === Collect all audio files ===
files = []
for root, dirs, fs in os.walk(input_dir):
    for f in fs:
        fname_lower = f.lower()
        if fname_lower.endswith(".wav") or fname_lower.endswith(".m4a") or ".m4a.wav" in fname_lower:
            files.append(os.path.join(root, f))

print(f"🔍 Found {len(files)} audio files")

# === Main Loop ===
results = []

for idx, file in enumerate(files, 1):
    fname = os.path.relpath(file, input_dir)

    try:
        noisy, sr = librosa.load(file, sr=16000, mono=True)
    except Exception as e:
        print(f"❌ Could not load {fname}: {e}")
        continue

    clean_sig = noisy  # same as noisy (no reference clean)

    # --- Metrics ---
    try:
        pesq_score = pesq(sr, noisy, clean_sig, "wb")
    except Exception:
        pesq_score = None

    snr_score = compute_snr(clean_sig, noisy)

    try:
        stoi_score = stoi(noisy, clean_sig, sr)
    except Exception:
        stoi_score = None

    quality = rate_quality(pesq_score, snr_score)

    results.append({
        "file": fname,
        "pesq": pesq_score,
        "snr": snr_score,
        "stoi": stoi_score,
        "quality": quality
    })

    print(f"[{idx}/{len(files)}] ✅ {fname} | PESQ={pesq_score}, SNR={snr_score}, STOI={stoi_score}, Quality={quality}")

    # Save progress every 20 files
    if idx % 20 == 0:
        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"💾 Progress saved at {idx} files")

# === Final Save ===
pd.DataFrame(results).to_csv(results_csv, index=False)
print(f"\n📂 Final results saved to {results_csv}")
