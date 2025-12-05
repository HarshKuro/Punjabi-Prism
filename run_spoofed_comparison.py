#!/usr/bin/env python3
"""
Spoofed vs Bonafide Comparison Pipeline
=======================================

Compares spoofed (replay attack) audio against bonafide (genuine) recordings.

For each spoofed file:
- Find the matching bonafide reference (same speaker, same sentence)
- Compute PESQ, SNR, STOI between spoofed and bonafide
- This measures how well spoofed audio matches original

Usage:
    python run_spoofed_comparison.py
    python run_spoofed_comparison.py --workers 8
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import DEFAULT_CONFIG, AudioConfig
from pipeline.preprocessing import AudioPreprocessor
from pipeline.alignment import AudioAligner
from pipeline.metrics import MetricsExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SpoofedFileInfo:
    """Metadata for a spoofed file"""
    filepath: str
    filename: str
    speaker_id: str
    sentence_id: str
    spoof_type: str  # Spoofed-1 or Spoofed-2
    recording_device: str
    distance: str
    original_device: str  # Device in filename (from original bonafide)


def parse_spoofed_filename(filepath: str) -> Optional[SpoofedFileInfo]:
    """
    Parse spoofed filename to extract metadata.
    
    Example: pa_S01_f1_female_IP14p_MBP15_0.5m_0_north_75db_1_S.wav
    """
    filename = os.path.basename(filepath)
    
    # Extract sentence ID
    sent_match = re.search(r'pa_S(\d+)', filename)
    if not sent_match:
        return None
    sentence_id = f"S{sent_match.group(1)}"
    
    # Extract speaker ID from filename
    speaker_match = re.search(r'_([fFmM]\d+)_', filename)
    if not speaker_match:
        return None
    speaker_id = speaker_match.group(1).lower()
    
    # Extract device from filename
    device_match = re.search(r'_(IP\w+|MBA\w*|MBP\w*)_', filename, re.IGNORECASE)
    original_device = device_match.group(1) if device_match else "unknown"
    
    # Extract spoof type and recording device from path
    parts = Path(filepath).parts
    spoof_type = None
    recording_device = "unknown"
    distance = "unknown"
    
    for i, part in enumerate(parts):
        if 'Spoofed-' in part:
            spoof_type = part
            # Next part should be distance/device folder
            if i + 1 < len(parts):
                folder_name = parts[i + 1]
                # Parse folder name like "0.5m iP14P" or "IP14 plus 1m"
                dist_match = re.search(r'(\d+\.?\d*)m', folder_name)
                if dist_match:
                    distance = dist_match.group(1) + "m"
                # Extract device from folder
                dev_match = re.search(r'(IP\d+\w*|MBA\w*|iPad\w*)', folder_name, re.IGNORECASE)
                if dev_match:
                    recording_device = dev_match.group(1)
    
    return SpoofedFileInfo(
        filepath=filepath,
        filename=filename,
        speaker_id=speaker_id,
        sentence_id=sentence_id,
        spoof_type=spoof_type or "unknown",
        recording_device=recording_device,
        distance=distance,
        original_device=original_device
    )


def find_bonafide_reference(spoofed_info: SpoofedFileInfo, 
                           bonafide_index: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Find matching bonafide file for a spoofed recording.
    
    Args:
        spoofed_info: Parsed spoofed file info
        bonafide_index: speaker -> sentence -> filepath mapping
        
    Returns:
        Path to bonafide reference or None
    """
    speaker = spoofed_info.speaker_id
    sentence = spoofed_info.sentence_id
    
    if speaker in bonafide_index and sentence in bonafide_index[speaker]:
        return bonafide_index[speaker][sentence]
    
    return None


def index_bonafide_files(bonafide_root: str) -> Dict[str, Dict[str, str]]:
    """
    Index bonafide files by speaker and sentence.
    
    Prefers 0m files as reference, then 0.5m, then minimum distance.
    
    Returns:
        Nested dict: speaker -> sentence -> best_reference_path
    """
    root = Path(bonafide_root)
    index = {}  # speaker -> sentence -> {distance: path}
    
    # First pass: collect all files
    temp_index = {}
    for wav_file in root.rglob("*.wav"):
        filename = wav_file.name
        
        # Parse speaker
        speaker_match = re.search(r'_([fFmM]\d+)_', filename)
        if not speaker_match:
            continue
        speaker = speaker_match.group(1).lower()
        
        # Parse sentence
        sent_match = re.search(r'pa_S(\d+)', filename)
        if not sent_match:
            continue
        sentence = f"S{sent_match.group(1)}"
        
        # Parse distance from folder
        folder_name = wav_file.parent.name
        dist_match = re.search(r'(\d+\.?\d*)m', folder_name)
        if dist_match:
            distance = float(dist_match.group(1))
        else:
            distance = 999  # Unknown distance, lowest priority
        
        if speaker not in temp_index:
            temp_index[speaker] = {}
        if sentence not in temp_index[speaker]:
            temp_index[speaker][sentence] = {}
        
        temp_index[speaker][sentence][distance] = str(wav_file)
    
    # Second pass: select best reference (prefer 0m, then lowest distance)
    for speaker, sentences in temp_index.items():
        index[speaker] = {}
        for sentence, distances in sentences.items():
            # Prefer 0m, then 0.5m, then minimum
            if 0 in distances:
                index[speaker][sentence] = distances[0]
            elif 0.5 in distances:
                index[speaker][sentence] = distances[0.5]
            else:
                min_dist = min(distances.keys())
                index[speaker][sentence] = distances[min_dist]
    
    return index


def process_single_comparison(args: Tuple[str, str, SpoofedFileInfo, AudioConfig]) -> Dict:
    """
    Process a single spoofed vs bonafide comparison.
    
    Args:
        args: (spoofed_path, bonafide_path, spoofed_info, config)
        
    Returns:
        Dictionary with comparison results
    """
    import time
    start_time = time.time()
    
    spoofed_path, bonafide_path, spoofed_info, config = args
    
    result = {
        "spoofed_file": os.path.basename(spoofed_path),
        "bonafide_file": os.path.basename(bonafide_path),
        "speaker_id": spoofed_info.speaker_id.upper(),
        "sentence_id": spoofed_info.sentence_id,
        "spoof_type": spoofed_info.spoof_type,
        "recording_device": spoofed_info.recording_device,
        "distance": spoofed_info.distance,
        "pesq": None,
        "snr_global_db": None,
        "snr_segmental_db": None,
        "stoi": None,
        "correlation": None,
        "alignment_shift_ms": None,
        "success": False,
        "error": None
    }
    
    try:
        preprocessor = AudioPreprocessor(config)
        aligner = AudioAligner(config)
        extractor = MetricsExtractor(config)
        
        # Preprocess both files
        bonafide_result = preprocessor.preprocess(bonafide_path)
        spoofed_result = preprocessor.preprocess(spoofed_path)
        
        if not bonafide_result.valid:
            result["error"] = f"Bonafide preprocessing failed: {bonafide_result.error}"
            return result
        
        if not spoofed_result.valid:
            result["error"] = f"Spoofed preprocessing failed: {spoofed_result.error}"
            return result
        
        # Align spoofed to bonafide
        alignment = aligner.align(
            bonafide_result.audio,
            spoofed_result.audio,
            config.SAMPLE_RATE,
            method='combined'
        )
        
        result["alignment_shift_ms"] = alignment.time_shift_ms
        result["correlation"] = alignment.correlation_score
        
        if not alignment.success:
            result["error"] = f"Alignment failed: {alignment.error}"
            return result
        
        # Extract metrics
        metrics = extractor.extract_all(
            alignment.reference,
            alignment.degraded,
            config.SAMPLE_RATE
        )
        
        if metrics.valid:
            result["pesq"] = metrics.pesq_score
            result["snr_global_db"] = metrics.snr_global_db
            result["snr_segmental_db"] = metrics.snr_segmental_db
            result["stoi"] = metrics.stoi_score
            result["success"] = True
        else:
            result["error"] = f"Metrics failed: {metrics.errors}"
        
    except Exception as e:
        result["error"] = str(e)
    
    result["processing_time_sec"] = time.time() - start_time
    return result


def run_spoofed_comparison(spoofed_root: str, 
                          bonafide_root: str,
                          output_dir: str,
                          num_workers: int = 4) -> pd.DataFrame:
    """
    Run full spoofed vs bonafide comparison.
    """
    logger.info("="*70)
    logger.info("SPOOFED vs BONAFIDE COMPARISON PIPELINE")
    logger.info("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Index bonafide files
    logger.info("Phase 1: Indexing bonafide files...")
    bonafide_index = index_bonafide_files(bonafide_root)
    total_bonafide = sum(len(s) for s in bonafide_index.values())
    logger.info(f"Indexed {total_bonafide} bonafide references across {len(bonafide_index)} speakers")
    
    # Find all spoofed files and match to bonafide
    logger.info("Phase 2: Scanning spoofed files...")
    spoofed_root_path = Path(spoofed_root)
    jobs = []
    
    for wav_file in spoofed_root_path.rglob("*.wav"):
        spoofed_info = parse_spoofed_filename(str(wav_file))
        if not spoofed_info:
            continue
        
        bonafide_ref = find_bonafide_reference(spoofed_info, bonafide_index)
        if not bonafide_ref:
            logger.warning(f"No bonafide reference for {wav_file.name}")
            continue
        
        jobs.append((str(wav_file), bonafide_ref, spoofed_info, DEFAULT_CONFIG.audio))
    
    logger.info(f"Created {len(jobs)} comparison jobs")
    
    # Process comparisons
    logger.info(f"Phase 3: Processing {len(jobs)} comparisons with {num_workers} workers...")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_comparison, job): job for job in jobs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Comparing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Job failed: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"spoofed_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Also save as latest
    latest_path = output_path / "spoofed_results.csv"
    df.to_csv(latest_path, index=False)
    
    # Summary
    successful = df["success"].sum()
    logger.info(f"\nPhase 4: Results Summary")
    logger.info(f"Total comparisons: {len(df)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(df) - successful}")
    
    if successful > 0:
        logger.info(f"\nMean PESQ: {df['pesq'].mean():.3f}")
        logger.info(f"Mean STOI: {df['stoi'].mean():.3f}")
        logger.info(f"Mean SNR: {df['snr_global_db'].mean():.2f} dB")
    
    logger.info(f"\nResults saved to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare Spoofed audio against Bonafide references"
    )
    parser.add_argument(
        "--spoofed", "-s",
        default="Spoofed",
        help="Path to Spoofed folder (default: Spoofed)"
    )
    parser.add_argument(
        "--bonafide", "-b",
        default="Bonafide",
        help="Path to Bonafide folder (default: Bonafide)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results_spoofed",
        help="Output directory (default: results_spoofed)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    df = run_spoofed_comparison(
        args.spoofed,
        args.bonafide,
        args.output,
        args.workers
    )
    
    # Print summary by spoof type
    print("\n" + "="*70)
    print("SPOOFED vs BONAFIDE COMPARISON SUMMARY")
    print("="*70)
    
    if len(df) > 0 and df["success"].sum() > 0:
        print("\nBy Spoof Type:")
        summary = df.groupby("spoof_type")[["pesq", "stoi", "snr_global_db"]].agg(["mean", "std", "count"])
        print(summary.round(3).to_string())
        
        print("\nBy Recording Device:")
        summary2 = df.groupby("recording_device")[["pesq", "stoi"]].agg(["mean", "count"])
        print(summary2.round(3).to_string())
        
        print("\nBy Speaker:")
        summary3 = df.groupby("speaker_id")[["pesq", "stoi"]].agg(["mean", "count"])
        print(summary3.round(3).to_string())
    
    print(f"\nResults saved to: {args.output}/spoofed_results.csv")


if __name__ == "__main__":
    main()
