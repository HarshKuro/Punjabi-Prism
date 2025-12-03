# Samsung Prism - Punjabi Speech Quality Assessment Pipeline

A production-grade audio quality assessment pipeline for the Samsung Prism Punjabi Speech Dataset. This tool computes industry-standard metrics (PESQ, SNR, STOI) to evaluate speech quality degradation across different recording distances and devices.

## 🎯 Project Overview

This pipeline was developed for **Samsung PRISM (Preparing and Inspiring Student Minds)** research program to analyze Punjabi speech recordings captured at various distances using different mobile devices.

### Key Features

- **ITU-T P.862 PESQ** (Perceptual Evaluation of Speech Quality)
- **SNR** (Signal-to-Noise Ratio) - Global, Segmental, and A-weighted
- **STOI** (Short-Time Objective Intelligibility)
- **Automated Reference Detection** - Uses closest-distance recordings as reference
- **DTW Alignment** - Dynamic Time Warping for accurate audio synchronization
- **Parallel Processing** - Multi-core batch processing for large datasets
- **Comprehensive Reports** - CSV exports, heatmaps, and statistical analysis

## 📁 Dataset Structure

The pipeline expects the following folder structure:

```
dataset/
├── F1/                    # Female Speaker 1
│   └── 1m/               # 1 meter distance
│       └── *.wav
├── F2/                    # Female Speaker 2
│   ├── F21m/             # 1 meter distance (reference)
│   ├── F22m/             # 2 meter distance
│   └── F24m/             # 4 meter distance
├── F3/                    # Female Speaker 3
│   └── F30m/             # 0 meter distance (reference quality)
├── M1/                    # Male Speaker 1
│   └── ...
└── M2/                    # Male Speaker 2
    └── M20m/             # 0 meter distance
```

### Filename Convention

Audio files follow this naming pattern:
```
pa_S01_f2_female_IP14p_na_1m_90_east_57db_0_B.wav
│  │   │  │      │     │  │  │  │    │    │ │
│  │   │  │      │     │  │  │  │    │    │ └── Channel
│  │   │  │      │     │  │  │  │    │    └──── Index
│  │   │  │      │     │  │  │  │    └───────── Noise level (dB)
│  │   │  │      │     │  │  │  └────────────── Direction
│  │   │  │      │     │  │  └───────────────── Angle
│  │   │  │      │     │  └──────────────────── Distance (meters)
│  │   │  │      │     └─────────────────────── Condition
│  │   │  │      └───────────────────────────── Device (IP14p, IP15, etc.)
│  │   │  └──────────────────────────────────── Gender
│  │   └─────────────────────────────────────── Speaker ID
│  └─────────────────────────────────────────── Sentence ID
└────────────────────────────────────────────── Language (Punjabi)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/HarshKuro/Punjabi-Prism.git
cd Punjabi-Prism

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Process all audio files in current directory
python run_pipeline.py

# Specify dataset location and output directory
python run_pipeline.py --dataset ./my_audio_data --output ./results

# Use more workers for faster processing
python run_pipeline.py --workers 8

# Quick test on first 10 files
python run_pipeline.py --test

# Verbose mode for debugging
python run_pipeline.py --verbose
```

### 3. View Results

After processing, find your results in the output directory:

```
results/
├── latest_results.csv              # Main results file
├── audio_quality_metrics_*.csv     # Timestamped results
├── summary_statistics_*.json       # Statistical summary
└── reports/
    ├── summary_report.txt          # Text summary
    ├── dashboard.html              # Interactive HTML dashboard
    ├── heatmap_pesq_distance_speaker.png
    ├── heatmap_snr_distance_speaker.png
    ├── quality_by_distance.png
    ├── metric_distributions.png
    ├── speaker_comparison.png
    └── correlation_matrix.png
```

## 📊 Output Metrics

### CSV Columns

| Column | Description |
|--------|-------------|
| `filename` | Degraded audio filename |
| `reference_filename` | Reference audio filename |
| `sentence_id` | Sentence identifier (S01-S50) |
| `speaker_id` | Speaker identifier (F1, F2, F3, M1, M2) |
| `gender` | Speaker gender |
| `device` | Recording device (IP14p, IP15, IP16p, IPDp11) |
| `distance_m` | Recording distance in meters |
| `noise_level_db` | Ambient noise level |
| `pesq` | PESQ score (1.0-4.5, higher is better) |
| `snr_global_db` | Global SNR in dB |
| `snr_segmental_db` | Segmental SNR in dB |
| `snr_aweighted_db` | A-weighted SNR in dB |
| `stoi` | STOI score (0-1, higher is better) |
| `correlation` | Correlation with reference |
| `alignment_shift_ms` | Time alignment shift in ms |
| `success` | Processing success status |

### Understanding PESQ Scores

| Score Range | Quality |
|-------------|---------|
| 4.0 - 4.5 | Excellent |
| 3.0 - 4.0 | Good |
| 2.0 - 3.0 | Fair |
| 1.0 - 2.0 | Poor |

> **Note:** Self-comparison (same file as reference) yields scores ~4.64, indicating near-perfect quality.

## 🔧 Advanced Usage

### Generate Reports from Existing Results

```bash
python run_pipeline.py --report-only results/latest_results.csv
```

### Skip Report Generation

```bash
python run_pipeline.py --no-report
```

### Python API

```python
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import DEFAULT_CONFIG

# Initialize orchestrator
orchestrator = PipelineOrchestrator(DEFAULT_CONFIG)

# Run pipeline
results = orchestrator.run(
    dataset_dir="./my_audio_data",
    output_dir="./results",
    num_workers=4
)

# Access results
print(f"Processed: {len(results)} files")
print(f"Success rate: {sum(r.success for r in results) / len(results) * 100:.1f}%")
```

### Custom Configuration

```python
from pipeline.config import PipelineConfig, AudioConfig, MetricsConfig

# Create custom config
config = PipelineConfig(
    audio=AudioConfig(
        SAMPLE_RATE=16000,
        SILENCE_THRESHOLD_DB=-40,
        RMS_TARGET_DB=-20
    ),
    metrics=MetricsConfig(
        COMPUTE_PESQ=True,
        COMPUTE_STOI=True,
        MIN_DURATION_SEC=0.5
    )
)
```

## 📋 Requirements

- Python 3.10+
- NumPy
- SciPy
- librosa
- pesq
- pystoi
- pandas
- matplotlib
- seaborn
- tqdm

See `requirements.txt` for complete list.

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE ORCHESTRATOR                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Reference Discovery                                          │
│     └── Scan folders → Find min-distance files → Index by       │
│         speaker/sentence                                         │
├─────────────────────────────────────────────────────────────────┤
│  2. Job Creation                                                 │
│     └── Create (reference, degraded) pairs for each distance    │
├─────────────────────────────────────────────────────────────────┤
│  3. Parallel Processing (multiprocessing)                        │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  For each job:                                           │ │
│     │  ├── Preprocessing (normalize, mono, trim, RMS)         │ │
│     │  ├── DTW Alignment (cross-correlation + DTW)            │ │
│     │  └── Metrics Extraction (PESQ, SNR, STOI)               │ │
│     └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  4. Results Aggregation                                          │
│     └── CSV export + JSON summary + visualizations              │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Punjabi-Prism/
├── run_pipeline.py          # Main entry point
├── pipeline/
│   ├── __init__.py
│   ├── config.py            # Configuration and file parsing
│   ├── preprocessing.py     # Audio preprocessing
│   ├── alignment.py         # DTW alignment
│   ├── metrics.py           # PESQ, SNR, STOI computation
│   ├── orchestrator.py      # Pipeline orchestration
│   └── reporting.py         # Visualization and reports
├── F1/, F2/, F3/, M1/, M2/  # Audio data folders
├── results/                  # Output directory
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

### Preprocessing Pipeline

1. **Sample Rate Normalization**: Resample to 16 kHz (required for PESQ)
2. **Mono Conversion**: Convert stereo to mono
3. **Silence Trimming**: Remove leading/trailing silence
4. **RMS Leveling**: Normalize to -20 dB RMS

### Alignment Algorithm

1. **Cross-correlation**: Find coarse time shift
2. **Shift Capping**: Limit shift to ±500ms to prevent artifacts
3. **DTW Refinement**: MFCC-based Dynamic Time Warping for fine alignment

### Reference Selection Strategy

- **Speakers with 0m recordings**: Use 0m as reference (highest quality)
- **Speakers with no 0m**: Use minimum distance as reference
- **Single-distance speakers**: Self-comparison for baseline metrics

## 📈 Expected Results

Based on Samsung Prism Punjabi dataset analysis:

| Distance | Avg PESQ | Avg SNR | Avg STOI |
|----------|----------|---------|----------|
| 0m (ref) | ~4.64 | ~70 dB | ~0.99 |
| 1m | ~4.64 | ~70 dB | ~0.99 |
| 2m | ~1.32 | ~-2.7 dB | ~0.68 |
| 4m | ~1.29 | ~-2.4 dB | ~0.65 |

> Lower scores at higher distances indicate expected quality degradation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-metric`)
3. Commit changes (`git commit -am 'Add new metric'`)
4. Push to branch (`git push origin feature/new-metric`)
5. Create Pull Request

## 📄 License

This project is part of the Samsung PRISM research program.

## 👥 Authors

- **Harsh Jain** - [HarshKuro](https://github.com/HarshKuro)

## 🙏 Acknowledgments

- Samsung PRISM Program for research support
- ITU-T for PESQ standard (P.862)
- librosa developers for audio processing tools

---

**Samsung PRISM** | Punjabi Speech Quality Assessment | 2025
