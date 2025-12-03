"""
Metrics Extraction Module
=========================

ITU-T P.862 compliant PESQ and validated SNR computation.

All metrics are computed with proper alignment and validation.

Features:
- PESQ (Wideband) using ITU-T P.862 standard
- Multiple SNR methods (global, segmental, A-weighted)
- STOI (Short-Time Objective Intelligibility)
- Comprehensive validation and error handling
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .config import AudioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Import quality metrics packages with graceful fallbacks
try:
    from pesq import pesq as compute_pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    logger.warning("pesq package not installed. Install with: pip install pesq")

try:
    from pystoi import stoi as compute_stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    logger.warning("pystoi package not installed. Install with: pip install pystoi")


@dataclass
class MetricsResult:
    """Complete metrics result for a single audio pair"""
    # Core metrics
    pesq_score: Optional[float] = None
    snr_global_db: Optional[float] = None
    snr_segmental_db: Optional[float] = None
    stoi_score: Optional[float] = None
    
    # Additional metrics
    snr_aweighted_db: Optional[float] = None
    correlation: Optional[float] = None
    energy_ratio_db: Optional[float] = None
    
    # Validation
    valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    
    # Metadata
    reference_duration: float = 0.0
    degraded_duration: float = 0.0
    aligned_duration: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "pesq": self.pesq_score,
            "snr_global": self.snr_global_db,
            "snr_segmental": self.snr_segmental_db,
            "snr_aweighted": self.snr_aweighted_db,
            "stoi": self.stoi_score,
            "correlation": self.correlation,
            "energy_ratio_db": self.energy_ratio_db,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "reference_duration": self.reference_duration,
            "degraded_duration": self.degraded_duration,
            "aligned_duration": self.aligned_duration
        }


class MetricsExtractor:
    """
    Extract quality metrics from aligned audio pairs.
    
    Implements ITU-T P.862 PESQ and validated SNR computation.
    """
    
    def __init__(self, config: AudioConfig = None):
        """
        Initialize metrics extractor.
        
        Args:
            config: AudioConfig instance
        """
        self.config = config or DEFAULT_CONFIG.audio
        
        # Check available metrics
        self.pesq_available = PESQ_AVAILABLE
        self.stoi_available = STOI_AVAILABLE
        
        if not self.pesq_available:
            logger.warning("PESQ computation unavailable - install pesq package")
        if not self.stoi_available:
            logger.warning("STOI computation unavailable - install pystoi package")
        
        logger.info(f"MetricsExtractor initialized (PESQ={self.pesq_available}, STOI={self.stoi_available})")
    
    # =========================================================================
    # SNR COMPUTATION
    # =========================================================================
    
    def compute_snr_global(self, reference: np.ndarray, 
                           degraded: np.ndarray) -> float:
        """
        Compute global SNR.
        
        SNR = 10 * log10(signal_power / noise_power)
        where noise = degraded - reference
        
        Args:
            reference: Clean reference signal
            degraded: Degraded signal (aligned)
            
        Returns:
            SNR in dB
        """
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        # Compute noise (error signal)
        noise = deg - ref
        
        # Compute powers
        signal_power = np.sum(ref ** 2)
        noise_power = np.sum(noise ** 2)
        
        # Handle edge cases
        if signal_power < 1e-10:
            logger.warning("Reference signal has near-zero power")
            return -np.inf
        
        if noise_power < 1e-10:
            logger.debug("Noise power near zero - signals nearly identical")
            return 100.0  # Cap at 100 dB
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        # Cap at reasonable range
        snr = np.clip(snr, -20, 100)
        
        return float(snr)
    
    def compute_snr_segmental(self, reference: np.ndarray,
                               degraded: np.ndarray,
                               sr: int) -> float:
        """
        Compute segmental (frame-by-frame) SNR.
        
        More robust than global SNR for speech signals.
        
        Args:
            reference: Clean reference signal
            degraded: Degraded signal (aligned)
            sr: Sample rate
            
        Returns:
            Segmental SNR in dB (averaged across frames)
        """
        # Frame parameters from config
        frame_size = int(self.config.SNR_FRAME_SIZE_MS * sr / 1000)
        hop_size = int(self.config.SNR_HOP_SIZE_MS * sr / 1000)
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        # Compute frame-by-frame SNR
        snr_values = []
        
        for start in range(0, len(ref) - frame_size, hop_size):
            end = start + frame_size
            
            ref_frame = ref[start:end]
            deg_frame = deg[start:end]
            noise_frame = deg_frame - ref_frame
            
            # Frame powers
            signal_power = np.sum(ref_frame ** 2)
            noise_power = np.sum(noise_frame ** 2)
            
            # Skip silent frames
            if signal_power < 1e-10:
                continue
            
            if noise_power < 1e-10:
                frame_snr = 35.0  # Cap for very clean frames
            else:
                frame_snr = 10 * np.log10(signal_power / noise_power)
            
            # Clip to reasonable range
            frame_snr = np.clip(frame_snr, -10, 35)
            snr_values.append(frame_snr)
        
        if not snr_values:
            logger.warning("No valid frames for segmental SNR")
            return 0.0
        
        # Return average
        return float(np.mean(snr_values))
    
    def compute_snr_aweighted(self, reference: np.ndarray,
                               degraded: np.ndarray,
                               sr: int) -> float:
        """
        Compute A-weighted SNR.
        
        A-weighting approximates human hearing sensitivity.
        
        Args:
            reference: Clean reference signal
            degraded: Degraded signal
            sr: Sample rate
            
        Returns:
            A-weighted SNR in dB
        """
        from scipy import signal as scipy_signal
        
        # Design A-weighting filter
        # Using simplified A-weighting approximation
        b, a = self._design_aweighting_filter(sr)
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        # Apply A-weighting
        ref_weighted = scipy_signal.lfilter(b, a, ref)
        deg_weighted = scipy_signal.lfilter(b, a, deg)
        
        # Compute SNR on weighted signals
        noise_weighted = deg_weighted - ref_weighted
        
        signal_power = np.sum(ref_weighted ** 2)
        noise_power = np.sum(noise_weighted ** 2)
        
        if signal_power < 1e-10 or noise_power < 1e-10:
            return self.compute_snr_global(reference, degraded)
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, -20, 100))
    
    def _design_aweighting_filter(self, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design A-weighting filter coefficients.
        
        Simplified 2nd order approximation.
        """
        from scipy import signal as scipy_signal
        
        # A-weighting pole frequencies
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        
        # Design high-pass at f1 (2nd order)
        # and low-pass at f4 (2nd order)
        # This is a simplified approximation
        
        # High-pass
        wn_hp = f1 / (sr / 2)
        wn_hp = min(wn_hp, 0.99)
        b_hp, a_hp = scipy_signal.butter(2, wn_hp, btype='highpass')
        
        # Low-pass
        wn_lp = f4 / (sr / 2)
        wn_lp = min(wn_lp, 0.99)
        b_lp, a_lp = scipy_signal.butter(2, wn_lp, btype='lowpass')
        
        # Combine filters
        b = np.convolve(b_hp, b_lp)
        a = np.convolve(a_hp, a_lp)
        
        return b, a
    
    # =========================================================================
    # PESQ COMPUTATION
    # =========================================================================
    
    def compute_pesq(self, reference: np.ndarray,
                     degraded: np.ndarray,
                     sr: int,
                     mode: str = None) -> Optional[float]:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality).
        
        ITU-T P.862 compliant implementation.
        
        Args:
            reference: Clean reference signal
            degraded: Degraded signal (aligned)
            sr: Sample rate (must be 8000 or 16000)
            mode: 'nb' (narrowband) or 'wb' (wideband)
            
        Returns:
            PESQ score (1.0 to 4.5) or None if computation fails
        """
        if not self.pesq_available:
            logger.warning("PESQ not available")
            return None
        
        mode = mode or self.config.PESQ_MODE
        
        # Validate sample rate
        if sr not in [8000, 16000]:
            logger.error(f"PESQ requires 8kHz or 16kHz, got {sr}Hz")
            return None
        
        # Use wideband mode for 16kHz
        if sr == 16000:
            mode = 'wb'
        elif sr == 8000:
            mode = 'nb'
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        # Validate duration
        duration = min_len / sr
        if duration < self.config.MIN_DURATION_SEC:
            logger.warning(f"Audio too short for PESQ: {duration:.2f}s")
            return None
        
        # Truncate if too long
        if duration > self.config.MAX_DURATION_SEC:
            max_samples = int(self.config.MAX_DURATION_SEC * sr)
            ref = ref[:max_samples]
            deg = deg[:max_samples]
            logger.debug(f"Truncated audio to {self.config.MAX_DURATION_SEC}s for PESQ")
        
        try:
            # Compute PESQ
            score = compute_pesq(sr, ref, deg, mode)
            
            # Validate score range (4.5 is max for degraded, but self-comparison can exceed)
            if score > 4.64:  # Self-comparison typically gives ~4.64
                logger.debug(f"PESQ score {score:.2f} indicates near-identical audio (self-comparison)")
            elif not (1.0 <= score <= 4.5):
                logger.warning(f"PESQ score out of range: {score}")
            
            return float(score)
            
        except Exception as e:
            logger.error(f"PESQ computation failed: {e}")
            return None
    
    # =========================================================================
    # STOI COMPUTATION
    # =========================================================================
    
    def compute_stoi(self, reference: np.ndarray,
                     degraded: np.ndarray,
                     sr: int,
                     extended: bool = False) -> Optional[float]:
        """
        Compute STOI (Short-Time Objective Intelligibility).
        
        Args:
            reference: Clean reference signal
            degraded: Degraded signal (aligned)
            sr: Sample rate
            extended: Use extended STOI (better for very degraded signals)
            
        Returns:
            STOI score (0.0 to 1.0) or None if computation fails
        """
        if not self.stoi_available:
            logger.warning("STOI not available")
            return None
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        try:
            # Compute STOI
            score = compute_stoi(ref, deg, sr, extended=extended)
            
            # Validate range
            score = float(np.clip(score, 0.0, 1.0))
            
            return score
            
        except Exception as e:
            logger.error(f"STOI computation failed: {e}")
            return None
    
    # =========================================================================
    # ADDITIONAL METRICS
    # =========================================================================
    
    def compute_correlation(self, reference: np.ndarray,
                           degraded: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            reference: Reference signal
            degraded: Degraded signal
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        corr = np.corrcoef(ref, deg)[0, 1]
        
        if np.isnan(corr):
            return 0.0
        
        return float(corr)
    
    def compute_energy_ratio(self, reference: np.ndarray,
                             degraded: np.ndarray) -> float:
        """
        Compute energy ratio in dB.
        
        Args:
            reference: Reference signal
            degraded: Degraded signal
            
        Returns:
            Energy ratio in dB (positive = degraded is louder)
        """
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        ref_energy = np.sum(ref ** 2)
        deg_energy = np.sum(deg ** 2)
        
        if ref_energy < 1e-10:
            return 0.0
        
        ratio_db = 10 * np.log10(deg_energy / ref_energy)
        return float(ratio_db)
    
    # =========================================================================
    # COMPLETE METRICS EXTRACTION
    # =========================================================================
    
    def extract_all(self, reference: np.ndarray,
                    degraded: np.ndarray,
                    sr: int,
                    compute_stoi: bool = True) -> MetricsResult:
        """
        Extract all quality metrics from aligned audio pair.
        
        Args:
            reference: Clean reference signal (preprocessed)
            degraded: Degraded signal (preprocessed and aligned)
            sr: Sample rate
            compute_stoi: Whether to compute STOI
            
        Returns:
            MetricsResult with all computed metrics
        """
        result = MetricsResult()
        
        # Store durations
        result.reference_duration = len(reference) / sr
        result.degraded_duration = len(degraded) / sr
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        result.aligned_duration = min_len / sr
        
        # Validate minimum duration
        if result.aligned_duration < self.config.MIN_DURATION_SEC:
            result.valid = False
            result.errors.append(f"Audio too short: {result.aligned_duration:.2f}s")
            return result
        
        # =====================================================================
        # SNR Metrics
        # =====================================================================
        
        try:
            result.snr_global_db = self.compute_snr_global(ref, deg)
            logger.debug(f"Global SNR: {result.snr_global_db:.2f} dB")
        except Exception as e:
            result.warnings.append(f"Global SNR failed: {e}")
        
        try:
            result.snr_segmental_db = self.compute_snr_segmental(ref, deg, sr)
            logger.debug(f"Segmental SNR: {result.snr_segmental_db:.2f} dB")
        except Exception as e:
            result.warnings.append(f"Segmental SNR failed: {e}")
        
        try:
            result.snr_aweighted_db = self.compute_snr_aweighted(ref, deg, sr)
            logger.debug(f"A-weighted SNR: {result.snr_aweighted_db:.2f} dB")
        except Exception as e:
            result.warnings.append(f"A-weighted SNR failed: {e}")
        
        # =====================================================================
        # PESQ
        # =====================================================================
        
        try:
            result.pesq_score = self.compute_pesq(ref, deg, sr)
            if result.pesq_score is not None:
                logger.debug(f"PESQ: {result.pesq_score:.3f}")
            else:
                result.warnings.append("PESQ returned None")
        except Exception as e:
            result.warnings.append(f"PESQ failed: {e}")
        
        # =====================================================================
        # STOI
        # =====================================================================
        
        if compute_stoi:
            try:
                result.stoi_score = self.compute_stoi(ref, deg, sr)
                if result.stoi_score is not None:
                    logger.debug(f"STOI: {result.stoi_score:.3f}")
            except Exception as e:
                result.warnings.append(f"STOI failed: {e}")
        
        # =====================================================================
        # Additional Metrics
        # =====================================================================
        
        try:
            result.correlation = self.compute_correlation(ref, deg)
            result.energy_ratio_db = self.compute_energy_ratio(ref, deg)
        except Exception as e:
            result.warnings.append(f"Additional metrics failed: {e}")
        
        # Final validation
        if result.pesq_score is None and result.snr_global_db is None:
            result.valid = False
            result.errors.append("No valid metrics computed")
        
        if result.warnings:
            logger.warning(f"Metrics warnings: {result.warnings}")
        
        return result


if __name__ == "__main__":
    # Test metrics extraction
    logging.basicConfig(level=logging.DEBUG)
    
    extractor = MetricsExtractor()
    
    # Create test signals
    sr = 16000
    t = np.linspace(0, 2, 2 * sr)
    
    # Reference: clean sine wave
    ref = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Degraded: add noise
    noise_level = 0.1
    deg = ref + noise_level * np.random.randn(len(ref))
    
    # Extract metrics
    result = extractor.extract_all(ref, deg, sr)
    
    print("\nMetrics Result:")
    print(f"  SNR Global: {result.snr_global_db:.2f} dB")
    print(f"  SNR Segmental: {result.snr_segmental_db:.2f} dB")
    print(f"  PESQ: {result.pesq_score}")
    print(f"  STOI: {result.stoi_score}")
    print(f"  Correlation: {result.correlation:.4f}")
    print(f"  Valid: {result.valid}")
