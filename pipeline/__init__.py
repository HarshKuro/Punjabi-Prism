"""
Samsung Prism - Punjabi Audio Quality Assessment Pipeline
=========================================================

Production-grade, deterministic audio scoring pipeline for
SNR and PESQ analysis across 2000+ WAV files.

Modules:
- config: Frozen preprocessing parameters and pipeline settings
- preprocessing: Audio normalization, silence trimming, RMS leveling
- alignment: DTW-based reference-degraded alignment
- metrics: ITU-T P.862 PESQ and validated SNR computation
- orchestrator: Multiprocessing batch processor
- reporting: CSV output and visualization dashboard
"""

__version__ = "1.0.0"
__author__ = "Samsung Prism Team"
