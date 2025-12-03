"""
Audio Preprocessing Module
==========================

Deterministic audio preprocessing for reproducible quality assessment.

Features:
- Sample rate normalization (16kHz for PESQ)
- Mono conversion
- Silence trimming (configurable threshold)
- RMS/Peak normalization
- Audio validation

All operations are deterministic and logged.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from pathlib import Path

from .config import AudioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of audio preprocessing"""
    audio: np.ndarray
    sample_rate: int
    original_duration: float
    processed_duration: float
    original_rms_db: float
    processed_rms_db: float
    peak_normalized: bool
    samples_trimmed: int
    valid: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "original_duration": self.original_duration,
            "processed_duration": self.processed_duration,
            "original_rms_db": self.original_rms_db,
            "processed_rms_db": self.processed_rms_db,
            "peak_normalized": self.peak_normalized,
            "samples_trimmed": self.samples_trimmed,
            "valid": self.valid,
            "error": self.error
        }


class AudioPreprocessor:
    """
    Deterministic audio preprocessor for quality assessment.
    
    All parameters are frozen in config for reproducibility.
    """
    
    def __init__(self, config: AudioConfig = None):
        """
        Initialize preprocessor with frozen config.
        
        Args:
            config: AudioConfig instance (uses DEFAULT if None)
        """
        self.config = config or DEFAULT_CONFIG.audio
        logger.info(f"AudioPreprocessor initialized with config v{self.config.CONFIG_VERSION}")
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with target sample rate and mono conversion.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load with librosa (handles resampling)
            audio, sr = librosa.load(
                filepath,
                sr=self.config.SAMPLE_RATE,
                mono=self.config.MONO
            )
            
            logger.debug(f"Loaded {filepath}: {len(audio)/sr:.2f}s @ {sr}Hz")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def compute_rms_db(self, audio: np.ndarray) -> float:
        """
        Compute RMS level in dB.
        
        Args:
            audio: Audio signal array
            
        Returns:
            RMS level in dB (relative to 1.0)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -np.inf
    
    def normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to peak amplitude of 1.0.
        
        Args:
            audio: Input audio array
            
        Returns:
            Peak-normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio
    
    def normalize_rms(self, audio: np.ndarray, target_db: float = None) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Args:
            audio: Input audio array
            target_db: Target RMS in dB (uses config default if None)
            
        Returns:
            RMS-normalized audio
        """
        target_db = target_db or self.config.TARGET_RMS_DB
        
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            target_rms = 10 ** (target_db / 20)
            gain = target_rms / current_rms
            
            # Apply gain with headroom protection
            normalized = audio * gain
            
            # Soft clipping to prevent harsh clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 0.99:
                normalized = normalized * (0.99 / max_val)
                logger.debug(f"Applied soft limiting: max={max_val:.3f}")
            
            return normalized
        
        return audio
    
    def trim_silence(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            Tuple of (trimmed_audio, samples_trimmed)
        """
        original_length = len(audio)
        
        # Use librosa's trim function with config threshold
        trimmed, (start_idx, end_idx) = librosa.effects.trim(
            audio,
            top_db=self.config.TOP_DB,
            frame_length=int(self.config.SNR_FRAME_SIZE_MS * sr / 1000),
            hop_length=int(self.config.SNR_HOP_SIZE_MS * sr / 1000)
        )
        
        samples_trimmed = original_length - len(trimmed)
        
        if samples_trimmed > 0:
            logger.debug(f"Trimmed {samples_trimmed} samples ({samples_trimmed/sr*1000:.1f}ms)")
        
        return trimmed, samples_trimmed
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> Tuple[bool, Optional[str]]:
        """
        Validate audio for PESQ computation.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        duration = len(audio) / sr
        
        # Check duration
        if duration < self.config.MIN_DURATION_SEC:
            return False, f"Audio too short: {duration:.2f}s < {self.config.MIN_DURATION_SEC}s"
        
        if duration > self.config.MAX_DURATION_SEC:
            return False, f"Audio too long: {duration:.2f}s > {self.config.MAX_DURATION_SEC}s"
        
        # Check for silence
        rms_db = self.compute_rms_db(audio)
        if rms_db < -60:
            return False, f"Audio nearly silent: RMS={rms_db:.1f}dB"
        
        # Check for clipping
        peak = np.max(np.abs(audio))
        if peak >= 1.0:
            clipping_samples = np.sum(np.abs(audio) >= 0.999)
            clipping_pct = 100 * clipping_samples / len(audio)
            if clipping_pct > 1.0:
                return False, f"Excessive clipping: {clipping_pct:.1f}% samples"
        
        # Check sample rate
        if sr != self.config.SAMPLE_RATE:
            return False, f"Wrong sample rate: {sr}Hz != {self.config.SAMPLE_RATE}Hz"
        
        return True, None
    
    def preprocess(self, filepath: str, 
                   trim_silence: bool = True,
                   normalize: bool = True) -> PreprocessingResult:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to audio file
            trim_silence: Whether to trim silence
            normalize: Whether to normalize audio
            
        Returns:
            PreprocessingResult with processed audio
        """
        try:
            # Load audio
            audio, sr = self.load_audio(filepath)
            original_duration = len(audio) / sr
            original_rms = self.compute_rms_db(audio)
            
            samples_trimmed = 0
            peak_normalized = False
            
            # Trim silence
            if trim_silence:
                audio, samples_trimmed = self.trim_silence(audio, sr)
            
            # Normalize
            if normalize:
                if self.config.PEAK_NORMALIZATION:
                    audio = self.normalize_peak(audio)
                    peak_normalized = True
                
                audio = self.normalize_rms(audio)
            
            processed_duration = len(audio) / sr
            processed_rms = self.compute_rms_db(audio)
            
            # Validate
            is_valid, error = self.validate_audio(audio, sr)
            
            return PreprocessingResult(
                audio=audio,
                sample_rate=sr,
                original_duration=original_duration,
                processed_duration=processed_duration,
                original_rms_db=original_rms,
                processed_rms_db=processed_rms,
                peak_normalized=peak_normalized,
                samples_trimmed=samples_trimmed,
                valid=is_valid,
                error=error
            )
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {filepath}: {e}")
            return PreprocessingResult(
                audio=np.array([]),
                sample_rate=self.config.SAMPLE_RATE,
                original_duration=0,
                processed_duration=0,
                original_rms_db=-np.inf,
                processed_rms_db=-np.inf,
                peak_normalized=False,
                samples_trimmed=0,
                valid=False,
                error=str(e)
            )
    
    def save_audio(self, audio: np.ndarray, sr: int, filepath: str):
        """
        Save processed audio to file.
        
        Args:
            audio: Audio array
            sr: Sample rate
            filepath: Output path
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with soundfile
        sf.write(filepath, audio, sr, subtype='PCM_16')
        logger.debug(f"Saved audio to {filepath}")


class BatchPreprocessor:
    """
    Batch preprocessing with caching and progress tracking.
    """
    
    def __init__(self, config: AudioConfig = None, cache_dir: str = None):
        """
        Initialize batch preprocessor.
        
        Args:
            config: AudioConfig instance
            cache_dir: Directory for caching preprocessed audio
        """
        self.preprocessor = AudioPreprocessor(config)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = {}
    
    def preprocess_batch(self, filepaths: list, 
                         use_cache: bool = True,
                         show_progress: bool = True) -> Dict[str, PreprocessingResult]:
        """
        Preprocess multiple files.
        
        Args:
            filepaths: List of file paths
            use_cache: Whether to use cached results
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping filepath to PreprocessingResult
        """
        from tqdm import tqdm
        
        results = {}
        iterator = tqdm(filepaths, desc="Preprocessing") if show_progress else filepaths
        
        for filepath in iterator:
            # Check cache
            if use_cache and filepath in self._cache:
                results[filepath] = self._cache[filepath]
                continue
            
            # Process
            result = self.preprocessor.preprocess(filepath)
            results[filepath] = result
            
            # Cache
            if use_cache:
                self._cache[filepath] = result
        
        # Summary
        valid_count = sum(1 for r in results.values() if r.valid)
        logger.info(f"Preprocessed {len(results)} files: {valid_count} valid, {len(results)-valid_count} invalid")
        
        return results


if __name__ == "__main__":
    # Test preprocessing
    logging.basicConfig(level=logging.DEBUG)
    
    preprocessor = AudioPreprocessor()
    print(f"Preprocessor config: SR={preprocessor.config.SAMPLE_RATE}Hz, PESQ={preprocessor.config.PESQ_MODE}")
