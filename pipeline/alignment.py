"""
Audio Alignment Module
======================

Dynamic Time Warping (DTW) for reference-degraded audio alignment.

Critical for accurate PESQ scoring - misaligned audio inflates errors.

Features:
- DTW-based alignment using librosa/dtw
- Configurable Sakoe-Chiba band constraint
- Cross-correlation pre-alignment for efficiency
- Alignment quality metrics
"""

import numpy as np
import librosa
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.interpolate import interp1d

from .config import AudioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of audio alignment"""
    reference: np.ndarray
    degraded: np.ndarray
    sample_rate: int
    alignment_path: Optional[np.ndarray]
    time_shift_samples: int
    time_shift_ms: float
    correlation_score: float
    dtw_cost: Optional[float]
    aligned_length: int
    method: str  # 'xcorr', 'dtw', 'combined'
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "time_shift_samples": self.time_shift_samples,
            "time_shift_ms": self.time_shift_ms,
            "correlation_score": self.correlation_score,
            "dtw_cost": self.dtw_cost,
            "aligned_length": self.aligned_length,
            "method": self.method,
            "success": self.success,
            "error": self.error
        }


class AudioAligner:
    """
    Audio alignment using cross-correlation and DTW.
    
    Ensures reference and degraded audio are temporally aligned
    for accurate PESQ computation.
    """
    
    def __init__(self, config: AudioConfig = None):
        """
        Initialize aligner with config.
        
        Args:
            config: AudioConfig instance
        """
        self.config = config or DEFAULT_CONFIG.audio
        logger.info("AudioAligner initialized")
    
    def cross_correlate(self, reference: np.ndarray, 
                        degraded: np.ndarray,
                        sr: int = 16000,
                        max_shift_ms: float = 500.0) -> Tuple[int, float]:
        """
        Find time shift using cross-correlation.
        
        Fast method for initial alignment before DTW.
        
        Args:
            reference: Reference audio array
            degraded: Degraded audio array
            sr: Sample rate for max_shift calculation
            max_shift_ms: Maximum allowed shift in milliseconds
            
        Returns:
            Tuple of (shift_samples, correlation_score)
        """
        # Calculate max shift in samples
        max_shift_samples = int(max_shift_ms * sr / 1000)
        
        # Use scipy's correlate for efficiency
        correlation = signal.correlate(degraded, reference, mode='full')
        
        # Find peak
        peak_idx = np.argmax(correlation)
        
        # Convert to shift (negative = degraded is ahead)
        shift = peak_idx - len(reference) + 1
        
        # Normalize correlation score
        norm_factor = np.sqrt(np.sum(reference**2) * np.sum(degraded**2))
        if norm_factor > 0:
            corr_score = correlation[peak_idx] / norm_factor
        else:
            corr_score = 0.0
        
        # Cap shift to maximum allowed
        original_shift = shift
        if abs(shift) > max_shift_samples:
            logger.warning(f"Shift {shift} exceeds max {max_shift_samples}, capping")
            shift = max(min(shift, max_shift_samples), -max_shift_samples)
        
        logger.debug(f"Cross-correlation: original_shift={original_shift}, capped_shift={shift} samples, score={corr_score:.4f}")
        
        return shift, corr_score
    
    def apply_shift(self, reference: np.ndarray, 
                    degraded: np.ndarray,
                    shift: int,
                    min_length_samples: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time shift and align lengths using zero-padding.
        
        Uses padding instead of trimming to preserve audio content.
        
        Args:
            reference: Reference audio
            degraded: Degraded audio
            shift: Shift in samples (positive = degraded is delayed)
            min_length_samples: Minimum required output length (default 0.5s at 16kHz)
            
        Returns:
            Tuple of (aligned_reference, aligned_degraded)
        """
        ref_len = len(reference)
        deg_len = len(degraded)
        
        # For very low shifts, just match lengths
        if abs(shift) < 100:  # Less than ~6ms at 16kHz
            min_len = min(ref_len, deg_len)
            return reference[:min_len], degraded[:min_len]
        
        # Use padding approach instead of trimming
        if shift > 0:
            # Degraded is delayed - pad start of reference or trim start of degraded
            # Choose approach that preserves more audio
            if shift < deg_len // 2:
                # Small shift: trim degraded start
                degraded = degraded[shift:]
                min_len = min(len(reference), len(degraded))
                reference = reference[:min_len]
                degraded = degraded[:min_len]
            else:
                # Large shift: pad reference start, keep degraded
                reference = np.pad(reference, (shift, 0), mode='constant')
                min_len = min(len(reference), len(degraded))
                reference = reference[:min_len]
                degraded = degraded[:min_len]
        else:
            # Reference is delayed (shift < 0)
            abs_shift = abs(shift)
            if abs_shift < ref_len // 2:
                # Small shift: trim reference start
                reference = reference[abs_shift:]
                min_len = min(len(reference), len(degraded))
                reference = reference[:min_len]
                degraded = degraded[:min_len]
            else:
                # Large shift: pad degraded start, keep reference
                degraded = np.pad(degraded, (abs_shift, 0), mode='constant')
                min_len = min(len(reference), len(degraded))
                reference = reference[:min_len]
                degraded = degraded[:min_len]
        
        # Safety check: ensure we have enough audio
        if len(reference) < min_length_samples:
            logger.warning(f"Aligned audio too short ({len(reference)} samples), using original lengths")
            min_len = min(ref_len, deg_len)
            return reference[:min_len] if len(reference) >= min_len else np.zeros(0), \
                   degraded[:min_len] if len(degraded) >= min_len else np.zeros(0)
        
        return reference, degraded
    
    def dtw_align(self, reference: np.ndarray, 
                  degraded: np.ndarray,
                  sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Perform DTW alignment using MFCCs.
        
        Uses MFCC features for more robust alignment than raw waveform.
        
        Args:
            reference: Reference audio
            degraded: Degraded audio
            sr: Sample rate
            
        Returns:
            Tuple of (aligned_ref, aligned_deg, path, cost)
        """
        try:
            # Compute MFCCs for alignment (more robust than raw waveform)
            hop_length = int(sr * 0.01)  # 10ms hop
            n_mfcc = 13
            
            mfcc_ref = librosa.feature.mfcc(
                y=reference, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
            )
            mfcc_deg = librosa.feature.mfcc(
                y=degraded, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
            )
            
            # Compute DTW path
            D, wp = librosa.sequence.dtw(
                X=mfcc_ref, Y=mfcc_deg,
                metric='euclidean',
                backtrack=True
            )
            
            # DTW cost (normalized)
            cost = D[-1, -1] / len(wp)
            
            # Convert frame path to sample path
            wp_samples = wp * hop_length
            
            # Warp degraded signal to match reference timing
            aligned_deg = self._warp_signal(degraded, wp, hop_length)
            
            # Ensure same length
            min_len = min(len(reference), len(aligned_deg))
            aligned_ref = reference[:min_len]
            aligned_deg = aligned_deg[:min_len]
            
            logger.debug(f"DTW alignment: cost={cost:.4f}, path_len={len(wp)}")
            
            return aligned_ref, aligned_deg, wp, cost
            
        except Exception as e:
            logger.warning(f"DTW alignment failed: {e}")
            # Fall back to simple length matching
            min_len = min(len(reference), len(degraded))
            return reference[:min_len], degraded[:min_len], None, float('inf')
    
    def _warp_signal(self, signal: np.ndarray, 
                     path: np.ndarray,
                     hop_length: int) -> np.ndarray:
        """
        Warp signal according to DTW path.
        
        Args:
            signal: Input signal to warp
            path: DTW path (frame indices) - NOTE: librosa returns this in reverse order
            hop_length: Hop length used for MFCC
            
        Returns:
            Warped signal
        """
        # IMPORTANT: librosa.sequence.dtw returns path in reverse order (end to start)
        # Reverse it to get forward order (start to end)
        path = path[::-1]
        
        # Convert path to sample indices
        # path[:, 0] = reference frame indices
        # path[:, 1] = signal frame indices
        
        ref_frames = path[:, 0]
        sig_frames = path[:, 1]
        
        # Build sample-level mapping
        ref_samples = ref_frames * hop_length
        sig_samples = sig_frames * hop_length
        
        # Target length based on reference
        target_len = int(ref_samples[-1])
        
        # Create interpolation function
        # Map from reference time to signal time
        valid_mask = sig_samples < len(signal)
        if not np.any(valid_mask):
            return signal
        
        ref_valid = ref_samples[valid_mask]
        sig_valid = sig_samples[valid_mask]
        
        # Create warped signal
        warped = np.zeros(target_len)
        
        # Simple nearest-neighbor warping
        for i, (r, s) in enumerate(zip(ref_valid[:-1], sig_valid[:-1])):
            r_next = ref_valid[i+1] if i+1 < len(ref_valid) else target_len
            s_next = sig_valid[i+1] if i+1 < len(sig_valid) else len(signal)
            
            r, r_next = int(r), int(r_next)
            s, s_next = int(s), int(s_next)
            
            # Copy samples with linear interpolation
            if r_next > r and s_next > s:
                src_len = s_next - s
                dst_len = r_next - r
                
                if src_len > 0 and dst_len > 0 and s_next <= len(signal):
                    src_samples = signal[s:s_next]
                    if dst_len == src_len:
                        warped[r:r_next] = src_samples
                    else:
                        # Resample
                        indices = np.linspace(0, len(src_samples)-1, dst_len)
                        warped[r:r_next] = np.interp(indices, np.arange(len(src_samples)), src_samples)
        
        return warped
    
    def align(self, reference: np.ndarray,
              degraded: np.ndarray,
              sr: int,
              method: str = 'combined') -> AlignmentResult:
        """
        Align degraded audio to reference.
        
        Args:
            reference: Reference (clean) audio
            degraded: Degraded audio to align
            sr: Sample rate
            method: 'xcorr', 'dtw', or 'combined'
            
        Returns:
            AlignmentResult with aligned audio
        """
        try:
            if method == 'xcorr':
                return self._align_xcorr(reference, degraded, sr)
            elif method == 'dtw':
                return self._align_dtw(reference, degraded, sr)
            else:  # combined
                return self._align_combined(reference, degraded, sr)
                
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            return AlignmentResult(
                reference=reference,
                degraded=degraded,
                sample_rate=sr,
                alignment_path=None,
                time_shift_samples=0,
                time_shift_ms=0,
                correlation_score=0,
                dtw_cost=None,
                aligned_length=min(len(reference), len(degraded)),
                method=method,
                success=False,
                error=str(e)
            )
    
    def _align_xcorr(self, reference: np.ndarray,
                     degraded: np.ndarray,
                     sr: int) -> AlignmentResult:
        """Cross-correlation only alignment"""
        shift, corr_score = self.cross_correlate(reference, degraded, sr=sr)
        aligned_ref, aligned_deg = self.apply_shift(reference, degraded, shift)
        
        # Check for alignment failure (empty arrays)
        if len(aligned_ref) == 0 or len(aligned_deg) == 0:
            logger.warning("Alignment produced empty arrays, falling back to length matching")
            min_len = min(len(reference), len(degraded))
            aligned_ref = reference[:min_len]
            aligned_deg = degraded[:min_len]
            shift = 0
            corr_score = 0.0
        
        return AlignmentResult(
            reference=aligned_ref,
            degraded=aligned_deg,
            sample_rate=sr,
            alignment_path=None,
            time_shift_samples=shift,
            time_shift_ms=shift * 1000 / sr,
            correlation_score=corr_score,
            dtw_cost=None,
            aligned_length=len(aligned_ref),
            method='xcorr',
            success=len(aligned_ref) > 0
        )
    
    def _align_dtw(self, reference: np.ndarray,
                   degraded: np.ndarray,
                   sr: int) -> AlignmentResult:
        """DTW only alignment"""
        aligned_ref, aligned_deg, path, cost = self.dtw_align(reference, degraded, sr)
        
        return AlignmentResult(
            reference=aligned_ref,
            degraded=aligned_deg,
            sample_rate=sr,
            alignment_path=path,
            time_shift_samples=0,
            time_shift_ms=0,
            correlation_score=0,
            dtw_cost=cost,
            aligned_length=len(aligned_ref),
            method='dtw',
            success=True
        )
    
    def _align_combined(self, reference: np.ndarray,
                        degraded: np.ndarray,
                        sr: int) -> AlignmentResult:
        """Combined: cross-correlation first, then DTW refinement"""
        # Step 1: Coarse alignment with cross-correlation
        shift, corr_score = self.cross_correlate(reference, degraded, sr=sr)
        ref_shifted, deg_shifted = self.apply_shift(reference, degraded, shift)
        
        # Check for alignment failure
        if len(ref_shifted) == 0 or len(deg_shifted) == 0:
            logger.warning("Cross-correlation produced empty arrays, using original with length matching")
            min_len = min(len(reference), len(degraded))
            ref_shifted = reference[:min_len]
            deg_shifted = degraded[:min_len]
            shift = 0
            corr_score = 0.0
        
        # Step 2: Fine alignment with DTW if correlation is poor AND we have enough audio
        dtw_cost = None
        path = None
        min_dtw_samples = sr  # Need at least 1 second for DTW
        
        if corr_score < 0.8 and len(ref_shifted) > min_dtw_samples:
            logger.debug(f"Low correlation ({corr_score:.3f}), applying DTW refinement")
            try:
                aligned_ref, aligned_deg, path, dtw_cost = self.dtw_align(
                    ref_shifted, deg_shifted, sr
                )
            except Exception as e:
                logger.warning(f"DTW failed: {e}, using cross-correlation only")
                aligned_ref, aligned_deg = ref_shifted, deg_shifted
        else:
            aligned_ref, aligned_deg = ref_shifted, deg_shifted
        
        return AlignmentResult(
            reference=aligned_ref,
            degraded=aligned_deg,
            sample_rate=sr,
            alignment_path=path,
            time_shift_samples=shift,
            time_shift_ms=shift * 1000 / sr,
            correlation_score=corr_score,
            dtw_cost=dtw_cost,
            aligned_length=len(aligned_ref),
            method='combined',
            success=True
        )
    
    def compute_alignment_quality(self, reference: np.ndarray,
                                  degraded: np.ndarray) -> Dict:
        """
        Compute alignment quality metrics.
        
        Args:
            reference: Reference audio
            degraded: Degraded audio (should be aligned)
            
        Returns:
            Dictionary with quality metrics
        """
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        ref = reference[:min_len]
        deg = degraded[:min_len]
        
        # Correlation
        corr = np.corrcoef(ref, deg)[0, 1]
        
        # Energy ratio
        ref_energy = np.sum(ref ** 2)
        deg_energy = np.sum(deg ** 2)
        energy_ratio_db = 10 * np.log10(deg_energy / ref_energy) if ref_energy > 0 else 0
        
        # Cross-correlation peak
        xcorr = signal.correlate(deg, ref, mode='same')
        xcorr_peak_idx = np.argmax(xcorr)
        xcorr_offset = xcorr_peak_idx - len(ref) // 2
        
        return {
            "correlation": corr,
            "energy_ratio_db": energy_ratio_db,
            "xcorr_offset_samples": xcorr_offset,
            "length_samples": min_len
        }


if __name__ == "__main__":
    # Test alignment
    logging.basicConfig(level=logging.DEBUG)
    
    aligner = AudioAligner()
    
    # Create test signals
    sr = 16000
    t = np.linspace(0, 1, sr)
    ref = np.sin(2 * np.pi * 440 * t)  # 440Hz sine
    
    # Create delayed + noisy version
    delay_samples = 100
    deg = np.concatenate([np.zeros(delay_samples), ref[:-delay_samples]])
    deg += np.random.randn(len(deg)) * 0.1
    
    # Align
    result = aligner.align(ref, deg, sr, method='xcorr')
    print(f"Alignment result: shift={result.time_shift_ms:.1f}ms, corr={result.correlation_score:.4f}")
