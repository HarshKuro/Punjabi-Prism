#!/usr/bin/env python3
"""
Spoofed-1 vs Spoofed-2 Direct Audio Comparison
==============================================

Compares Spoofed-1 recordings directly against Spoofed-2 recordings
for the same speaker and sentence.

This measures the difference between two replay attack methods.
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
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import DEFAULT_CONFIG, AudioConfig
from pipeline.preprocessing import AudioPreprocessor
from pipeline.alignment import AudioAligner
from pipeline.metrics import MetricsExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def parse_spoofed_file(filepath: str) -> Optional[Dict]:
    """Parse spoofed filename to extract metadata."""
    filename = os.path.basename(filepath)
    
    # Extract sentence ID
    sent_match = re.search(r'pa_S(\d+)', filename)
    if not sent_match:
        return None
    sentence_id = f"S{sent_match.group(1).zfill(2)}"
    
    # Extract speaker ID
    speaker_match = re.search(r'_([fFmM]\d+)_', filename)
    if not speaker_match:
        return None
    speaker_id = speaker_match.group(1).upper()
    
    # Get spoof type from path
    parts = Path(filepath).parts
    spoof_type = None
    for part in parts:
        if 'Spoofed-1' in part:
            spoof_type = 'Spoofed-1'
            break
        elif 'Spoofed-2' in part:
            spoof_type = 'Spoofed-2'
            break
    
    if not spoof_type:
        return None
    
    return {
        'filepath': filepath,
        'filename': filename,
        'speaker_id': speaker_id,
        'sentence_id': sentence_id,
        'spoof_type': spoof_type
    }


def index_spoofed_files(spoofed_root: str) -> Tuple[Dict, Dict]:
    """
    Index all spoofed files by speaker and sentence.
    
    Returns:
        spoof1_index: speaker -> sentence -> [filepaths]
        spoof2_index: speaker -> sentence -> [filepaths]
    """
    root = Path(spoofed_root)
    spoof1_index = {}
    spoof2_index = {}
    
    for wav_file in root.rglob("*.wav"):
        info = parse_spoofed_file(str(wav_file))
        if not info:
            continue
        
        speaker = info['speaker_id']
        sentence = info['sentence_id']
        spoof_type = info['spoof_type']
        
        if spoof_type == 'Spoofed-1':
            if speaker not in spoof1_index:
                spoof1_index[speaker] = {}
            if sentence not in spoof1_index[speaker]:
                spoof1_index[speaker][sentence] = []
            spoof1_index[speaker][sentence].append(str(wav_file))
        else:
            if speaker not in spoof2_index:
                spoof2_index[speaker] = {}
            if sentence not in spoof2_index[speaker]:
                spoof2_index[speaker][sentence] = []
            spoof2_index[speaker][sentence].append(str(wav_file))
    
    return spoof1_index, spoof2_index


def process_comparison(args: Tuple[str, str, str, str, AudioConfig]) -> Dict:
    """Process a single Spoofed-1 vs Spoofed-2 comparison."""
    import time
    start_time = time.time()
    
    spoof1_path, spoof2_path, speaker_id, sentence_id, config = args
    
    result = {
        "spoof1_file": os.path.basename(spoof1_path),
        "spoof2_file": os.path.basename(spoof2_path),
        "speaker_id": speaker_id,
        "sentence_id": sentence_id,
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
        spoof1_result = preprocessor.preprocess(spoof1_path)
        spoof2_result = preprocessor.preprocess(spoof2_path)
        
        if not spoof1_result.valid:
            result["error"] = f"Spoof1 preprocessing failed: {spoof1_result.error}"
            return result
        
        if not spoof2_result.valid:
            result["error"] = f"Spoof2 preprocessing failed: {spoof2_result.error}"
            return result
        
        # Align spoof2 to spoof1 (spoof1 as reference)
        alignment = aligner.align(
            spoof1_result.audio,
            spoof2_result.audio,
            config.SAMPLE_RATE,
            method='combined'
        )
        
        result["alignment_shift_ms"] = alignment.time_shift_ms
        result["correlation"] = alignment.correlation_score
        
        if not alignment.success:
            result["error"] = f"Alignment failed: {alignment.error}"
            return result
        
        # Extract metrics (Spoof1 as reference, Spoof2 as degraded)
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


def run_spoof1_vs_spoof2(spoofed_root: str, output_dir: str, num_workers: int = 4):
    """Run Spoofed-1 vs Spoofed-2 direct comparison."""
    
    logger.info("="*70)
    logger.info("SPOOFED-1 vs SPOOFED-2 DIRECT AUDIO COMPARISON")
    logger.info("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Index files
    logger.info("Phase 1: Indexing spoofed files...")
    spoof1_index, spoof2_index = index_spoofed_files(spoofed_root)
    
    spoof1_count = sum(len(sentences) for sentences in spoof1_index.values())
    spoof2_count = sum(len(sentences) for sentences in spoof2_index.values())
    logger.info(f"Spoofed-1 files: {spoof1_count} across {len(spoof1_index)} speakers")
    logger.info(f"Spoofed-2 files: {spoof2_count} across {len(spoof2_index)} speakers")
    
    # Create comparison jobs (match by speaker and sentence)
    logger.info("Phase 2: Creating comparison pairs...")
    jobs = []
    
    for speaker in spoof1_index:
        if speaker not in spoof2_index:
            continue
        
        for sentence in spoof1_index[speaker]:
            if sentence not in spoof2_index[speaker]:
                continue
            
            # Get files for this speaker/sentence
            spoof1_files = spoof1_index[speaker][sentence]
            spoof2_files = spoof2_index[speaker][sentence]
            
            # Compare each spoof1 file with each spoof2 file
            for s1_file in spoof1_files:
                for s2_file in spoof2_files:
                    jobs.append((s1_file, s2_file, speaker, sentence, DEFAULT_CONFIG.audio))
    
    logger.info(f"Created {len(jobs)} comparison pairs")
    
    # Process comparisons
    logger.info(f"Phase 3: Processing {len(jobs)} comparisons with {num_workers} workers...")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_comparison, job): job for job in jobs}
        
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
    csv_path = output_path / f"spoof1_vs_spoof2_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    latest_path = output_path / "spoof1_vs_spoof2_results.csv"
    df.to_csv(latest_path, index=False)
    
    # Summary
    successful = df["success"].sum()
    logger.info(f"\nPhase 4: Results Summary")
    logger.info(f"Total comparisons: {len(df)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(df) - successful}")
    
    if successful > 0:
        print("\n" + "="*70)
        print("SPOOFED-1 vs SPOOFED-2 COMPARISON RESULTS")
        print("="*70)
        
        print(f"\nOverall Metrics (Spoof1 as reference, Spoof2 as test):")
        print(f"  Mean PESQ: {df['pesq'].mean():.3f} ± {df['pesq'].std():.3f}")
        print(f"  Mean STOI: {df['stoi'].mean():.3f} ± {df['stoi'].std():.3f}")
        print(f"  Mean SNR:  {df['snr_global_db'].mean():.2f} ± {df['snr_global_db'].std():.2f} dB")
        print(f"  Mean Correlation: {df['correlation'].mean():.3f}")
        
        print("\nBy Speaker:")
        speaker_stats = df.groupby('speaker_id')[['pesq', 'stoi', 'snr_global_db', 'correlation']].agg(['mean', 'count'])
        print(speaker_stats.round(3).to_string())
        
        print("\nPESQ Distribution:")
        print(f"  Min:  {df['pesq'].min():.3f}")
        print(f"  25%:  {df['pesq'].quantile(0.25):.3f}")
        print(f"  50%:  {df['pesq'].quantile(0.50):.3f}")
        print(f"  75%:  {df['pesq'].quantile(0.75):.3f}")
        print(f"  Max:  {df['pesq'].max():.3f}")
        
        print("\nSTOI Distribution:")
        print(f"  Min:  {df['stoi'].min():.3f}")
        print(f"  25%:  {df['stoi'].quantile(0.25):.3f}")
        print(f"  50%:  {df['stoi'].quantile(0.50):.3f}")
        print(f"  75%:  {df['stoi'].quantile(0.75):.3f}")
        print(f"  Max:  {df['stoi'].max():.3f}")
    
    logger.info(f"\nResults saved to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare Spoofed-1 audio directly against Spoofed-2 audio"
    )
    parser.add_argument(
        "--spoofed", "-s",
        default="Spoofed",
        help="Path to Spoofed folder (default: Spoofed)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results_spoof1_vs_spoof2",
        help="Output directory (default: results_spoof1_vs_spoof2)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    run_spoof1_vs_spoof2(args.spoofed, args.output, args.workers)


if __name__ == "__main__":
    main()
