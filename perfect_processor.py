"""
Perfect Audio Quality Processor
==============================

Main processing engine that orchestrates the complete audio quality analysis
with proper validation, uncertainty quantification, and statistical analysis.
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from perfect_audio_quality import (
    AudioValidator, StandardizedPESQ, ValidatedSNR, StandardizedSTOI,
    AudioMetadata, QualityMeasurement, ProcessingProvenance,
    logger, RANDOM_SEED
)

class ReferenceManager:
    """Manages reference audio files with proper validation"""
    
    def __init__(self, reference_directory: str):
        self.reference_directory = Path(reference_directory)
        self.reference_cache = {}
        self.reference_metadata = {}
        self.validator = AudioValidator()
        
    def load_and_validate_references(self) -> Dict[str, str]:
        """
        Load and validate all reference files
        
        Returns:
            Dictionary mapping speaker_id to reference file path
        """
        logger.info(f"Loading references from {self.reference_directory}")
        
        if not self.reference_directory.exists():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_directory}")
        
        reference_files = {}
        audio_extensions = ('.wav', '.mp3', '.m4a', '.flac')
        
        for file_path in self.reference_directory.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                # Extract speaker ID
                speaker_id = self._extract_speaker_id(file_path.name)
                if speaker_id:
                    # Validate reference quality
                    validation_result = self.validator.validate_audio_file(str(file_path))
                    
                    if validation_result['validation_passed']:
                        reference_files[speaker_id] = str(file_path)
                        self.reference_metadata[speaker_id] = validation_result
                        logger.info(f"✅ Reference validated: {speaker_id} -> {file_path.name}")
                    else:
                        logger.warning(f"❌ Reference validation failed: {file_path.name}")
                        for error in validation_result['errors']:
                            logger.warning(f"   Error: {error}")
        
        logger.info(f"Loaded {len(reference_files)} validated reference files")
        return reference_files
    
    def _extract_speaker_id(self, filename: str) -> Optional[str]:
        """Extract speaker ID from filename"""
        # Pattern for pa_S##_... format
        match = re.search(r'pa_(S\d+)_', filename)
        if match:
            return match.group(1)
        
        # Additional patterns can be added here
        return None
    
    def get_reference_quality_metrics(self, speaker_id: str) -> Optional[Dict]:
        """Get quality metrics for reference file"""
        return self.reference_metadata.get(speaker_id, {}).get('metrics')

class MetadataExtractor:
    """Extract and validate metadata from file paths and names"""
    
    def __init__(self):
        self.distance_patterns = [
            (r'(\d+)m', lambda m: int(m.group(1))),  # Direct: 1m, 2m, etc.
            (r'F(\d)(\d)m', lambda m: int(m.group(2))),  # F21m -> 1m, F22m -> 2m
            (r'F(\d+)0m', lambda m: 0),  # F30m -> 0m
            (r'M(\d)(\d)m', lambda m: int(m.group(2)))  # M20m -> 0m (assumed)
        ]
        
        self.gender_patterns = [
            (r'_f\d+_', 'female'),
            (r'_M\d+_', 'male'),
            (r'female', 'female'),
            (r'male', 'male')
        ]
    
    def extract_metadata(self, file_path: str) -> AudioMetadata:
        """
        Extract comprehensive metadata from file path
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioMetadata object with extracted information
        """
        file_path_obj = Path(file_path)
        filename = file_path_obj.name
        parent_dir = file_path_obj.parent.name
        
        # Extract speaker ID
        speaker_id = self._extract_speaker_id(filename)
        
        # Extract distance
        distance = self._extract_distance(parent_dir, filename)
        
        # Extract gender
        gender = self._extract_gender(filename)
        
        return AudioMetadata(
            file_path=file_path,
            speaker_id=speaker_id or "unknown",
            gender=gender or "unknown",
            distance_meters=distance,
            validated=False
        )
    
    def _extract_speaker_id(self, filename: str) -> Optional[str]:
        """Extract speaker ID from filename"""
        match = re.search(r'pa_(S\d+)_', filename)
        return match.group(1) if match else None
    
    def _extract_distance(self, parent_dir: str, filename: str) -> float:
        """Extract distance from folder name or filename"""
        # Try folder name first
        for pattern, extractor in self.distance_patterns:
            match = re.search(pattern, parent_dir)
            if match:
                try:
                    return float(extractor(match))
                except:
                    continue
        
        # Try filename
        for pattern, extractor in self.distance_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return float(extractor(match))
                except:
                    continue
        
        logger.warning(f"Could not extract distance from {parent_dir}/{filename}")
        return 0.0
    
    def _extract_gender(self, filename: str) -> Optional[str]:
        """Extract gender from filename"""
        for pattern, gender in self.gender_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return gender
        return None

class PerfectAudioProcessor:
    """
    Main processor orchestrating complete audio quality analysis
    """
    
    def __init__(self, reference_directory: str, output_directory: str = "results"):
        self.reference_manager = ReferenceManager(reference_directory)
        self.metadata_extractor = MetadataExtractor()
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize quality calculators
        self.pesq_calculator = StandardizedPESQ()
        self.snr_calculator = ValidatedSNR()
        self.stoi_calculator = StandardizedSTOI()
        self.validator = AudioValidator()
        
        # Load references
        self.references = self.reference_manager.load_and_validate_references()
        
        logger.info(f"Perfect Audio Processor initialized")
        logger.info(f"Available references: {list(self.references.keys())}")
    
    def process_single_file(self, file_path: str) -> Optional[Dict]:
        """
        Process a single audio file with complete quality analysis
        
        Args:
            file_path: Path to audio file to process
            
        Returns:
            Dictionary with complete analysis results
        """
        try:
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(file_path)
            
            # Validate audio file
            validation_result = self.validator.validate_audio_file(file_path)
            if not validation_result['validation_passed']:
                logger.warning(f"File validation failed: {file_path}")
                return None
            
            # Find matching reference
            reference_path = self.references.get(metadata.speaker_id)
            if not reference_path:
                logger.warning(f"No reference found for speaker {metadata.speaker_id}")
                return None
            
            # Load audio files
            import librosa
            degraded_audio, deg_sr = librosa.load(file_path, sr=None)
            reference_audio, ref_sr = librosa.load(reference_path, sr=None)
            
            # Calculate quality metrics
            results = {
                'file_path': file_path,
                'speaker_id': metadata.speaker_id,
                'gender': metadata.gender,
                'distance_meters': metadata.distance_meters,
                'reference_path': reference_path,
                'validation_metrics': validation_result['metrics']
            }
            
            # PESQ calculation
            pesq_result = self.pesq_calculator.calculate_pesq_with_uncertainty(
                reference_audio, degraded_audio, ref_sr
            )
            results['pesq'] = asdict(pesq_result)
            
            # SNR calculations
            snr_results = self.snr_calculator.calculate_snr_suite(
                reference_audio, degraded_audio, ref_sr
            )
            for snr_type, snr_measurement in snr_results.items():
                results[snr_type] = asdict(snr_measurement)
            
            # STOI calculation
            stoi_result = self.stoi_calculator.calculate_stoi_with_uncertainty(
                reference_audio, degraded_audio, ref_sr
            )
            results['stoi'] = asdict(stoi_result)
            
            # Additional metrics
            results['processing_metadata'] = {
                'degraded_sr': deg_sr,
                'reference_sr': ref_sr,
                'degraded_duration': len(degraded_audio) / deg_sr,
                'reference_duration': len(reference_audio) / ref_sr,
                'random_seed': RANDOM_SEED
            }
            
            logger.info(f"✅ Processed: {Path(file_path).name} | "
                       f"PESQ: {pesq_result.value:.3f}±{pesq_result.uncertainty:.3f} | "
                       f"SNR: {snr_results['global_snr'].value:.1f}±{snr_results['global_snr'].uncertainty:.1f}dB | "
                       f"STOI: {stoi_result.value:.3f}±{stoi_result.uncertainty:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_dataset(self, dataset_directory: str, max_workers: int = 4) -> pd.DataFrame:
        """
        Process entire dataset with parallel processing
        
        Args:
            dataset_directory: Directory containing audio files
            max_workers: Number of parallel workers
            
        Returns:
            DataFrame with all results
        """
        dataset_path = Path(dataset_directory)
        
        # Find all audio files
        audio_extensions = ('.wav', '.mp3', '.m4a', '.flac')
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(dataset_path.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(audio_files)} audio files in {dataset_path}")
        
        # Process files with parallel execution
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_single_file, str(file_path)): file_path
                for file_path in audio_files
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_file), 
                             total=len(future_to_file), 
                             desc="Processing audio files"):
                
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame
        if all_results:
            df = self._results_to_dataframe(all_results)
            
            # Save results
            output_file = self.output_directory / f"quality_analysis_{dataset_path.name}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Save detailed results as JSON
            json_file = self.output_directory / f"detailed_results_{dataset_path.name}.json"
            with open(json_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            return df
        else:
            logger.warning("No valid results obtained")
            return pd.DataFrame()
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results list to pandas DataFrame"""
        flattened_results = []
        
        for result in results:
            flat_result = {
                'file_path': result['file_path'],
                'speaker_id': result['speaker_id'],
                'gender': result['gender'],
                'distance_meters': result['distance_meters'],
                'reference_path': result['reference_path']
            }
            
            # Add validation metrics
            for key, value in result['validation_metrics'].items():
                flat_result[f'validation_{key}'] = value
            
            # Add PESQ
            pesq_data = result['pesq']
            flat_result['pesq_score'] = pesq_data['value']
            flat_result['pesq_uncertainty'] = pesq_data['uncertainty']
            flat_result['pesq_ci_lower'] = pesq_data['confidence_interval'][0]
            flat_result['pesq_ci_upper'] = pesq_data['confidence_interval'][1]
            
            # Add SNR metrics
            for snr_type in ['global_snr', 'segmental_snr', 'perceptual_snr']:
                if snr_type in result:
                    snr_data = result[snr_type]
                    flat_result[f'{snr_type}_score'] = snr_data['value']
                    flat_result[f'{snr_type}_uncertainty'] = snr_data['uncertainty']
                    flat_result[f'{snr_type}_ci_lower'] = snr_data['confidence_interval'][0]
                    flat_result[f'{snr_type}_ci_upper'] = snr_data['confidence_interval'][1]
            
            # Add STOI
            stoi_data = result['stoi']
            flat_result['stoi_score'] = stoi_data['value']
            flat_result['stoi_uncertainty'] = stoi_data['uncertainty']
            flat_result['stoi_ci_lower'] = stoi_data['confidence_interval'][0]
            flat_result['stoi_ci_upper'] = stoi_data['confidence_interval'][1]
            
            # Add processing metadata
            for key, value in result['processing_metadata'].items():
                flat_result[f'meta_{key}'] = value
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary report"""
        summary = {
            'dataset_summary': {
                'total_files': len(df),
                'unique_speakers': df['speaker_id'].nunique(),
                'distance_range': (df['distance_meters'].min(), df['distance_meters'].max()),
                'gender_distribution': df['gender'].value_counts().to_dict()
            },
            'quality_metrics_summary': {},
            'statistical_analysis': {}
        }
        
        # Quality metrics summary
        metrics = ['pesq_score', 'global_snr_score', 'stoi_score']
        for metric in metrics:
            if metric in df.columns:
                summary['quality_metrics_summary'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'median': float(df[metric].median())
                }
        
        # Distance-based analysis
        if 'distance_meters' in df.columns and 'pesq_score' in df.columns:
            distance_analysis = df.groupby('distance_meters').agg({
                'pesq_score': ['mean', 'std', 'count'],
                'global_snr_score': ['mean', 'std'],
                'stoi_score': ['mean', 'std']
            }).round(3)
            
            summary['distance_analysis'] = distance_analysis.to_dict()
        
        return summary

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Perfect Audio Quality Analysis')
    parser.add_argument('--dataset', required=True, help='Dataset directory path')
    parser.add_argument('--references', required=True, help='Reference files directory')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PerfectAudioProcessor(
        reference_directory=args.references,
        output_directory=args.output
    )
    
    # Process dataset
    results_df = processor.process_dataset(
        dataset_directory=args.dataset,
        max_workers=args.workers
    )
    
    if not results_df.empty:
        # Generate summary report
        summary = processor.generate_summary_report(results_df)
        
        # Save summary
        summary_file = Path(args.output) / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Processed {len(results_df)} files")
        logger.info(f"Results saved to {args.output}")
    else:
        logger.error("No valid results obtained")

if __name__ == "__main__":
    main()
