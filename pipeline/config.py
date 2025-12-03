"""
Pipeline Configuration Module
=============================

FROZEN preprocessing parameters and pipeline settings.
DO NOT MODIFY without version bump and documentation.

All settings are deterministic for reproducibility.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
import json
from datetime import datetime

# ============================================================================
# FROZEN PREPROCESSING PARAMETERS - DO NOT MODIFY
# ============================================================================

@dataclass(frozen=True)
class AudioConfig:
    """
    Frozen audio preprocessing configuration.
    
    All parameters are locked for reproducibility.
    Modify only with version increment and audit trail.
    """
    # Sample rate (ITU-T P.862 requires 8kHz or 16kHz for PESQ)
    SAMPLE_RATE: int = 16000
    
    # Audio format
    MONO: bool = True
    BIT_DEPTH: int = 16
    
    # Silence detection/trimming parameters
    SILENCE_THRESHOLD_DB: float = -40.0  # dB below peak
    SILENCE_MIN_DURATION_MS: int = 100   # Minimum silence duration to trim
    TOP_DB: int = 30                      # librosa.effects.trim parameter
    
    # RMS normalization
    TARGET_RMS_DB: float = -20.0         # Target RMS level
    PEAK_NORMALIZATION: bool = True      # Normalize to peak before RMS
    
    # DTW alignment parameters
    DTW_RADIUS: int = 1                  # Sakoe-Chiba band radius
    DTW_STEP_PATTERN: str = "symmetric2" # Step pattern for DTW
    
    # PESQ specific
    PESQ_MODE: str = "wb"                # Wide-band mode (16kHz)
    MIN_DURATION_SEC: float = 0.5        # Minimum duration for PESQ
    MAX_DURATION_SEC: float = 30.0       # Maximum duration for PESQ
    
    # SNR computation
    SNR_FRAME_SIZE_MS: int = 25          # Frame size for segmental SNR
    SNR_HOP_SIZE_MS: int = 10            # Hop size for segmental SNR
    SNR_VAD_THRESHOLD_DB: float = -35.0  # VAD threshold for SNR
    
    # Version tracking
    CONFIG_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class FolderConfig:
    """
    Folder naming conventions for Samsung Prism dataset.
    
    Expected structure:
    root/
    ├── F1/           # Speaker F1 (female)
    │   ├── 1m/       # 1 meter distance recordings
    │   ├── 2m/       # 2 meter distance recordings
    │   └── 0m/       # Reference recordings (clean)
    ├── F2/
    │   ├── F21m/     # Alternative naming: Speaker+Distance
    │   ├── F22m/
    │   └── F20m/
    ├── M1/           # Speaker M1 (male)
    │   └── ...
    └── M2/
        └── M20m/     # 0m = reference
    """
    # Speaker prefixes
    SPEAKER_PREFIXES: tuple = ("F", "M")
    
    # Distance identifiers (regex patterns)
    DISTANCE_PATTERN: str = r"(\d+)m"
    
    # Reference distance (used as clean baseline)
    REFERENCE_DISTANCE: str = "0m"
    
    # File pattern for Punjabi audio
    FILE_PATTERN: str = r"pa_S(\d+)_([FM]\d+)_(\w+)_(\w+)_(\w+)_(\d+)m_.*\.wav$"
    
    # Output directories
    OUTPUT_DIR: str = "pipeline_output"
    REPORTS_DIR: str = "reports"
    CACHE_DIR: str = ".pipeline_cache"
    LOGS_DIR: str = "logs"


@dataclass
class PipelineConfig:
    """
    Main pipeline configuration.
    
    Combines audio and folder configs with runtime settings.
    """
    audio: AudioConfig = field(default_factory=AudioConfig)
    folders: FolderConfig = field(default_factory=FolderConfig)
    
    # Runtime settings (can be modified)
    n_workers: int = 4                    # Parallel workers
    batch_size: int = 50                  # Files per batch
    verbose: bool = True                  # Detailed logging
    save_aligned_audio: bool = False      # Save DTW-aligned files
    compute_stoi: bool = True             # Also compute STOI
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Generate config hash for version tracking"""
        self._config_hash = self._compute_hash()
        self._created_at = datetime.now().isoformat()
    
    def _compute_hash(self) -> str:
        """Compute deterministic hash of frozen parameters"""
        config_dict = {
            "audio": {
                k: v for k, v in self.audio.__dict__.items()
                if not k.startswith("_")
            },
            "folders": {
                k: v for k, v in self.folders.__dict__.items()
                if not k.startswith("_")
            }
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    @property
    def config_hash(self) -> str:
        return self._config_hash
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return {
            "audio": {k: v for k, v in self.audio.__dict__.items()},
            "folders": {k: v for k, v in self.folders.__dict__.items()},
            "runtime": {
                "n_workers": self.n_workers,
                "batch_size": self.batch_size,
                "verbose": self.verbose,
                "save_aligned_audio": self.save_aligned_audio,
                "compute_stoi": self.compute_stoi,
                "random_seed": self.random_seed
            },
            "meta": {
                "config_hash": self._config_hash,
                "created_at": self._created_at,
                "version": self.audio.CONFIG_VERSION
            }
        }
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)
        
        config = cls()
        config.n_workers = data["runtime"]["n_workers"]
        config.batch_size = data["runtime"]["batch_size"]
        config.verbose = data["runtime"]["verbose"]
        config.save_aligned_audio = data["runtime"]["save_aligned_audio"]
        config.compute_stoi = data["runtime"]["compute_stoi"]
        config.random_seed = data["runtime"]["random_seed"]
        
        return config


# ============================================================================
# DATASET METADATA PARSER
# ============================================================================

@dataclass
class AudioFileMetadata:
    """Parsed metadata from Punjabi audio filename"""
    filepath: str
    filename: str
    sentence_id: str          # S01, S02, etc.
    speaker_id: str           # F1, F2, M1, M2, etc.
    gender: str               # male/female
    device: str               # IP14p, IP16p, etc.
    condition: str            # na, NA, etc.
    distance_m: int           # 0, 1, 2, 4, etc.
    noise_level_db: Optional[int] = None  # 50db, 57db, etc.
    direction: Optional[str] = None       # east, west, etc.
    angle: Optional[int] = None           # 90, etc.
    
    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "sentence_id": self.sentence_id,
            "speaker_id": self.speaker_id,
            "gender": self.gender,
            "device": self.device,
            "condition": self.condition,
            "distance_m": self.distance_m,
            "noise_level_db": self.noise_level_db,
            "direction": self.direction,
            "angle": self.angle
        }


def parse_filename(filepath: str) -> Optional[AudioFileMetadata]:
    """
    Parse Punjabi audio filename to extract metadata.
    
    Supports patterns:
    - pa_S01_f1_female_IP14p_na_1m_90_east_57db_0_B.wav
    - pa_S01_M2_male_IP16p_NA_0m_50db_0_B.wav
    
    Args:
        filepath: Path to audio file
        
    Returns:
        AudioFileMetadata or None if parsing fails
    """
    import re
    
    filename = os.path.basename(filepath)
    
    # Try pattern 1: Full format with direction/angle
    pattern1 = r"pa_S(\d+)_([fFmM]\d+)_(\w+)_(\w+)_(\w+)_(\d+)m_(\d+)_(\w+)_(\d+)db"
    match1 = re.search(pattern1, filename)
    
    if match1:
        return AudioFileMetadata(
            filepath=filepath,
            filename=filename,
            sentence_id=f"S{match1.group(1)}",
            speaker_id=match1.group(2).upper(),
            gender=match1.group(3).lower(),
            device=match1.group(4),
            condition=match1.group(5),
            distance_m=int(match1.group(6)),
            angle=int(match1.group(7)),
            direction=match1.group(8),
            noise_level_db=int(match1.group(9))
        )
    
    # Try pattern 2: Simplified format
    pattern2 = r"pa_S(\d+)_([fFmM]\d+)_(\w+)_(\w+)_(\w+)_(\d+)m_(\d+)db"
    match2 = re.search(pattern2, filename)
    
    if match2:
        return AudioFileMetadata(
            filepath=filepath,
            filename=filename,
            sentence_id=f"S{match2.group(1)}",
            speaker_id=match2.group(2).upper(),
            gender=match2.group(3).lower(),
            device=match2.group(4),
            condition=match2.group(5),
            distance_m=int(match2.group(6)),
            noise_level_db=int(match2.group(7))
        )
    
    # Try minimal pattern
    pattern3 = r"pa_S(\d+)_([fFmM]\d+)_(\w+)_(\w+)"
    match3 = re.search(pattern3, filename)
    
    if match3:
        # Extract distance from folder path
        distance = 0
        dist_match = re.search(r"(\d+)m", filepath)
        if dist_match:
            distance = int(dist_match.group(1))
        
        return AudioFileMetadata(
            filepath=filepath,
            filename=filename,
            sentence_id=f"S{match3.group(1)}",
            speaker_id=match3.group(2).upper(),
            gender=match3.group(3).lower(),
            device=match3.group(4),
            condition="unknown",
            distance_m=distance
        )
    
    return None


# ============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================================

DEFAULT_CONFIG = PipelineConfig()


if __name__ == "__main__":
    # Print configuration for verification
    config = PipelineConfig()
    print(f"Pipeline Configuration v{config.audio.CONFIG_VERSION}")
    print(f"Config Hash: {config.config_hash}")
    print(f"\nAudio Settings:")
    print(f"  Sample Rate: {config.audio.SAMPLE_RATE} Hz")
    print(f"  PESQ Mode: {config.audio.PESQ_MODE}")
    print(f"  Target RMS: {config.audio.TARGET_RMS_DB} dB")
    print(f"\nFolder Settings:")
    print(f"  Reference Distance: {config.folders.REFERENCE_DISTANCE}")
    print(f"  Output Directory: {config.folders.OUTPUT_DIR}")
