"""
Pipeline Orchestrator
=====================

Production-grade batch processing with multiprocessing support.

Features:
- Automatic reference discovery (0m = clean baseline)
- Parallel processing with worker pool
- Progress tracking and resumability
- Comprehensive logging and error handling
- Structured output generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np

from .config import (
    PipelineConfig, DEFAULT_CONFIG, AudioFileMetadata, 
    parse_filename, AudioConfig, FolderConfig
)
from .preprocessing import AudioPreprocessor, PreprocessingResult
from .alignment import AudioAligner, AlignmentResult
from .metrics import MetricsExtractor, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Single file processing job"""
    degraded_path: str
    reference_path: str
    metadata: AudioFileMetadata
    
    def to_dict(self) -> Dict:
        return {
            "degraded_path": self.degraded_path,
            "reference_path": self.reference_path,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


@dataclass
class ProcessingResult:
    """Result of processing a single file"""
    job: ProcessingJob
    preprocessing: Optional[PreprocessingResult] = None
    alignment: Optional[AlignmentResult] = None
    metrics: Optional[MetricsResult] = None
    success: bool = False
    error: Optional[str] = None
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        result = {
            "degraded_file": self.job.degraded_path,
            "reference_file": self.job.reference_path,
            "alignment_shift_ms": self.alignment.time_shift_ms if self.alignment else None,
            "alignment_correlation": self.alignment.correlation_score if self.alignment else None,
            "success": self.success,
            "error": self.error,
            "processing_time_sec": self.processing_time_sec
        }
        if self.job.metadata:
            result.update(self.job.metadata.to_dict())
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result
    
    def to_csv_row(self) -> Dict:
        """Flatten result for CSV export"""
        row = {
            "filename": os.path.basename(self.job.degraded_path),
            "reference_filename": os.path.basename(self.job.reference_path),
        }
        
        # Add metadata
        if self.job.metadata:
            row.update({
                "sentence_id": self.job.metadata.sentence_id,
                "speaker_id": self.job.metadata.speaker_id,
                "gender": self.job.metadata.gender,
                "device": self.job.metadata.device,
                "distance_m": self.job.metadata.distance_m,
                "noise_level_db": self.job.metadata.noise_level_db,
            })
        
        # Add metrics
        if self.metrics:
            row.update({
                "pesq": self.metrics.pesq_score,
                "snr_global_db": self.metrics.snr_global_db,
                "snr_segmental_db": self.metrics.snr_segmental_db,
                "snr_aweighted_db": self.metrics.snr_aweighted_db,
                "stoi": self.metrics.stoi_score,
                "correlation": self.metrics.correlation,
            })
        
        # Add alignment info
        if self.alignment:
            row.update({
                "alignment_shift_ms": self.alignment.time_shift_ms,
                "alignment_correlation": self.alignment.correlation_score,
            })
        
        row.update({
            "success": self.success,
            "error": self.error,
            "processing_time_sec": self.processing_time_sec,
        })
        
        return row


class ReferenceManager:
    """
    Manage reference (clean) audio files.
    
    Automatically discovers and maps references based on:
    - 0m distance recordings (closest to source)
    - Minimum distance if 0m not available
    - Same speaker, same sentence (or cross-speaker fallback)
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.references: Dict[str, Dict[str, str]] = {}  # speaker -> sentence -> filepath
        self.all_files: Dict[str, Dict[str, Dict[int, str]]] = {}  # speaker -> sentence -> distance -> filepath
        self._loaded = False
    
    def discover_references(self, root_dir: str, 
                           use_min_distance: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Auto-discover reference files.
        
        Args:
            root_dir: Root directory of dataset
            use_min_distance: If True, use minimum distance as reference when 0m not available
            
        Returns:
            Nested dict: speaker -> sentence -> reference_path
        """
        root = Path(root_dir)
        references = {}
        all_files = {}  # speaker -> sentence -> distance -> filepath
        
        # First pass: index ALL wav files with their distances
        for wav_file in root.rglob("*.wav"):
            metadata = parse_filename(str(wav_file))
            if not metadata:
                continue
            
            speaker_id = metadata.speaker_id.upper()
            sentence_id = metadata.sentence_id
            distance = metadata.distance_m
            
            if speaker_id not in all_files:
                all_files[speaker_id] = {}
            if sentence_id not in all_files[speaker_id]:
                all_files[speaker_id][sentence_id] = {}
            
            all_files[speaker_id][sentence_id][distance] = str(wav_file)
        
        self.all_files = all_files
        
        # Second pass: select references (prefer 0m, else minimum distance)
        for speaker_id, sentences in all_files.items():
            if speaker_id not in references:
                references[speaker_id] = {}
            
            for sentence_id, distances in sentences.items():
                if 0 in distances:
                    # Use 0m as reference
                    references[speaker_id][sentence_id] = distances[0]
                elif use_min_distance and distances:
                    # Use minimum distance as reference
                    min_dist = min(distances.keys())
                    references[speaker_id][sentence_id] = distances[min_dist]
                    logger.debug(f"Using {min_dist}m as reference for {speaker_id}/{sentence_id}")
        
        self.references = references
        self._loaded = True
        
        # Summary
        total_refs = sum(len(sents) for sents in references.values())
        logger.info(f"Discovered {total_refs} reference files across {len(references)} speakers")
        
        return references
    
    def get_reference(self, speaker_id: str, sentence_id: str) -> Optional[str]:
        """
        Get reference file path for a specific speaker and sentence.
        
        Args:
            speaker_id: Speaker ID (e.g., 'F1', 'M2')
            sentence_id: Sentence ID (e.g., 'S01')
            
        Returns:
            Path to reference file or None
        """
        speaker_id = speaker_id.upper()
        
        if speaker_id in self.references:
            if sentence_id in self.references[speaker_id]:
                return self.references[speaker_id][sentence_id]
        
        logger.warning(f"No reference found for {speaker_id}/{sentence_id}")
        return None
    
    def has_reference(self, speaker_id: str, sentence_id: str) -> bool:
        """Check if reference exists"""
        speaker_id = speaker_id.upper()
        return (speaker_id in self.references and 
                sentence_id in self.references[speaker_id])
    
    def get_distances_for_sentence(self, speaker_id: str, sentence_id: str) -> List[int]:
        """Get all available distances for a sentence"""
        speaker_id = speaker_id.upper()
        if speaker_id in self.all_files and sentence_id in self.all_files[speaker_id]:
            return sorted(self.all_files[speaker_id][sentence_id].keys())
        return []
    
    def get_reference_distance(self, speaker_id: str, sentence_id: str) -> Optional[int]:
        """Get the distance of the reference file"""
        speaker_id = speaker_id.upper()
        if speaker_id in self.all_files and sentence_id in self.all_files[speaker_id]:
            ref_path = self.get_reference(speaker_id, sentence_id)
            if ref_path:
                for dist, path in self.all_files[speaker_id][sentence_id].items():
                    if path == ref_path:
                        return dist
        return None


class DatasetScanner:
    """
    Scan dataset and build processing job queue.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def scan(self, root_dir: str, 
             reference_manager: ReferenceManager,
             min_distance_diff: int = 0,
             include_self_comparison: bool = True) -> List[ProcessingJob]:
        """
        Scan dataset and create processing jobs.
        
        Args:
            root_dir: Root directory of dataset
            reference_manager: Reference manager with discovered refs
            min_distance_diff: Minimum distance difference between ref and degraded
            include_self_comparison: If True, include reference files as self-comparison
                                    (useful for speakers with only one distance)
            
        Returns:
            List of ProcessingJob instances
        """
        jobs = []
        skipped = 0
        self_comparisons = 0
        
        # Use the reference manager's indexed files
        for speaker_id, sentences in reference_manager.all_files.items():
            for sentence_id, distances in sentences.items():
                # Get reference
                ref_path = reference_manager.get_reference(speaker_id, sentence_id)
                if not ref_path:
                    continue
                
                ref_dist = reference_manager.get_reference_distance(speaker_id, sentence_id)
                
                # Check if this speaker/sentence has only one distance
                has_only_one_distance = len(distances) == 1
                
                # Create jobs for all distances
                for distance, wav_path in distances.items():
                    # Parse metadata
                    metadata = parse_filename(wav_path)
                    if not metadata:
                        skipped += 1
                        continue
                    
                    # If same as reference
                    if wav_path == ref_path:
                        # Include self-comparison for single-distance speakers
                        if include_self_comparison and has_only_one_distance:
                            job = ProcessingJob(
                                degraded_path=wav_path,
                                reference_path=ref_path,
                                metadata=metadata
                            )
                            jobs.append(job)
                            self_comparisons += 1
                        continue
                    
                    # Skip if distance difference too small
                    if ref_dist is not None and abs(distance - ref_dist) < min_distance_diff:
                        continue
                    
                    # Create job
                    job = ProcessingJob(
                        degraded_path=wav_path,
                        reference_path=ref_path,
                        metadata=metadata
                    )
                    jobs.append(job)
        
        logger.info(f"Created {len(jobs)} processing jobs ({self_comparisons} self-comparisons), skipped {skipped} files")
        
        return jobs


def process_single_job(job: ProcessingJob, 
                       config: AudioConfig) -> ProcessingResult:
    """
    Process a single audio pair (worker function).
    
    This function is designed to be called in a separate process.
    
    Args:
        job: ProcessingJob with file paths
        config: AudioConfig instance
        
    Returns:
        ProcessingResult with all metrics
    """
    import time
    start_time = time.time()
    
    result = ProcessingResult(job=job)
    
    try:
        # Initialize components
        preprocessor = AudioPreprocessor(config)
        aligner = AudioAligner(config)
        extractor = MetricsExtractor(config)
        
        # 1. Preprocess both files
        ref_result = preprocessor.preprocess(job.reference_path)
        deg_result = preprocessor.preprocess(job.degraded_path)
        
        result.preprocessing = deg_result
        
        if not ref_result.valid or not deg_result.valid:
            result.error = f"Preprocessing failed: ref={ref_result.error}, deg={deg_result.error}"
            result.processing_time_sec = time.time() - start_time
            return result
        
        # 2. Align degraded to reference
        alignment = aligner.align(
            ref_result.audio, 
            deg_result.audio,
            config.SAMPLE_RATE,
            method='combined'
        )
        
        result.alignment = alignment
        
        if not alignment.success:
            result.error = f"Alignment failed: {alignment.error}"
            result.processing_time_sec = time.time() - start_time
            return result
        
        # 3. Extract metrics
        metrics = extractor.extract_all(
            alignment.reference,
            alignment.degraded,
            config.SAMPLE_RATE
        )
        
        result.metrics = metrics
        result.success = metrics.valid
        
        if not metrics.valid:
            result.error = f"Metrics extraction failed: {metrics.errors}"
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"Processing failed for {job.degraded_path}: {e}")
    
    result.processing_time_sec = time.time() - start_time
    return result


class PipelineOrchestrator:
    """
    Main pipeline orchestrator with multiprocessing support.
    
    Usage:
        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run("dataset_root/", output_dir="results/")
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize orchestrator.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config or DEFAULT_CONFIG
        self.reference_manager = ReferenceManager(self.config)
        self.scanner = DatasetScanner(self.config)
        
        # Results storage
        self.results: List[ProcessingResult] = []
        self._run_metadata: Dict = {}
    
    def run(self, dataset_dir: str,
            output_dir: str = None,
            n_workers: int = None,
            show_progress: bool = True) -> pd.DataFrame:
        """
        Run complete pipeline on dataset.
        
        Args:
            dataset_dir: Path to dataset root
            output_dir: Output directory for results
            n_workers: Number of parallel workers (None = use config)
            show_progress: Show progress bar
            
        Returns:
            DataFrame with all results
        """
        import time
        start_time = time.time()
        
        n_workers = n_workers or self.config.n_workers
        output_dir = output_dir or self.config.folders.OUTPUT_DIR
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store run metadata
        self._run_metadata = {
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
            "config_hash": self.config.config_hash,
            "config_version": self.config.audio.CONFIG_VERSION,
            "start_time": datetime.now().isoformat(),
            "n_workers": n_workers
        }
        
        logger.info(f"Starting pipeline run (config v{self.config.audio.CONFIG_VERSION})")
        logger.info(f"Dataset: {dataset_dir}")
        logger.info(f"Workers: {n_workers}")
        
        # 1. Discover references
        logger.info("Phase 1: Discovering references...")
        self.reference_manager.discover_references(dataset_dir)
        
        # 2. Scan dataset and create jobs
        logger.info("Phase 2: Scanning dataset...")
        jobs = self.scanner.scan(dataset_dir, self.reference_manager)
        
        if not jobs:
            logger.warning("No processing jobs found!")
            return pd.DataFrame()
        
        self._run_metadata["total_jobs"] = len(jobs)
        logger.info(f"Created {len(jobs)} processing jobs")
        
        # 3. Process jobs in parallel
        logger.info(f"Phase 3: Processing {len(jobs)} files...")
        self.results = self._process_parallel(jobs, n_workers, show_progress)
        
        # 4. Generate results
        logger.info("Phase 4: Generating results...")
        df = self._generate_dataframe()
        
        # 5. Save outputs
        self._save_results(df, output_path)
        
        # Summary
        elapsed = time.time() - start_time
        successful = sum(1 for r in self.results if r.success)
        
        self._run_metadata.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_sec": elapsed,
            "successful": successful,
            "failed": len(self.results) - successful
        })
        
        logger.info(f"Pipeline complete: {successful}/{len(self.results)} successful in {elapsed:.1f}s")
        
        return df
    
    def _process_parallel(self, jobs: List[ProcessingJob],
                          n_workers: int,
                          show_progress: bool) -> List[ProcessingResult]:
        """
        Process jobs in parallel using ProcessPoolExecutor.
        """
        results = []
        
        # Use multiprocessing for CPU-bound work
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(process_single_job, job, self.config.audio): job
                for job in jobs
            }
            
            # Collect results with progress bar
            iterator = as_completed(future_to_job)
            if show_progress:
                iterator = tqdm(iterator, total=len(jobs), desc="Processing")
            
            for future in iterator:
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results.append(result)
                except Exception as e:
                    job = future_to_job[future]
                    logger.error(f"Job failed: {job.degraded_path}: {e}")
                    results.append(ProcessingResult(
                        job=job,
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def _process_sequential(self, jobs: List[ProcessingJob],
                            show_progress: bool) -> List[ProcessingResult]:
        """
        Process jobs sequentially (for debugging).
        """
        results = []
        iterator = tqdm(jobs, desc="Processing") if show_progress else jobs
        
        for job in iterator:
            result = process_single_job(job, self.config.audio)
            results.append(result)
        
        return results
    
    def _generate_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        """
        rows = [r.to_csv_row() for r in self.results]
        df = pd.DataFrame(rows)
        
        # Sort by speaker, sentence, distance
        if 'speaker_id' in df.columns:
            df = df.sort_values(['speaker_id', 'sentence_id', 'distance_m'])
        
        return df
    
    def _save_results(self, df: pd.DataFrame, output_path: Path):
        """
        Save results to various formats.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Master CSV
        csv_path = output_path / f"audio_quality_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # 2. Summary statistics
        summary = self._compute_summary(df)
        summary_path = output_path / f"summary_statistics_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")
        
        # 3. Run metadata
        meta_path = output_path / f"run_metadata_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(self._run_metadata, f, indent=2, default=str)
        
        # 4. Config snapshot
        config_path = output_path / f"config_snapshot_{timestamp}.json"
        self.config.save(str(config_path))
        
        # 5. Create symlink to latest
        latest_csv = output_path / "latest_results.csv"
        if latest_csv.exists():
            latest_csv.unlink()
        try:
            latest_csv.symlink_to(csv_path.name)
        except (OSError, NotImplementedError):
            # Symlinks may not work on Windows
            df.to_csv(latest_csv, index=False)
    
    def _compute_summary(self, df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics from results.
        """
        summary = {
            "total_files": len(df),
            "successful": int(df['success'].sum()),
            "failed": int((~df['success']).sum()),
        }
        
        # Metrics summaries
        metrics = ['pesq', 'snr_global_db', 'snr_segmental_db', 'stoi']
        
        for metric in metrics:
            if metric in df.columns:
                valid = df[metric].dropna()
                if len(valid) > 0:
                    summary[f"{metric}_mean"] = float(valid.mean())
                    summary[f"{metric}_std"] = float(valid.std())
                    summary[f"{metric}_min"] = float(valid.min())
                    summary[f"{metric}_max"] = float(valid.max())
                    summary[f"{metric}_median"] = float(valid.median())
        
        # By distance
        if 'distance_m' in df.columns and 'pesq' in df.columns:
            by_distance = df.groupby('distance_m')['pesq'].agg(['mean', 'std', 'count'])
            summary['pesq_by_distance'] = by_distance.to_dict('index')
        
        # By speaker
        if 'speaker_id' in df.columns and 'pesq' in df.columns:
            by_speaker = df.groupby('speaker_id')['pesq'].agg(['mean', 'std', 'count'])
            summary['pesq_by_speaker'] = by_speaker.to_dict('index')
        
        return summary


def run_pipeline(dataset_dir: str,
                 output_dir: str = "pipeline_output",
                 n_workers: int = 4,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        dataset_dir: Path to dataset root directory
        output_dir: Output directory for results
        n_workers: Number of parallel workers
        verbose: Enable verbose logging
        
    Returns:
        DataFrame with all metrics
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/pipeline.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure pipeline
    config = PipelineConfig()
    config.n_workers = n_workers
    config.verbose = verbose
    
    # Run
    orchestrator = PipelineOrchestrator(config)
    df = orchestrator.run(dataset_dir, output_dir, n_workers)
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <dataset_dir> [output_dir] [n_workers]")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pipeline_output"
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    df = run_pipeline(dataset_dir, output_dir, n_workers)
    print(f"\nResults saved to {output_dir}/")
    print(df.head())
