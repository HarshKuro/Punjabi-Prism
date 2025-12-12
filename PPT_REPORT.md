# Samsung PRISM - Punjabi Speech Quality & Spoof Detection
## Comprehensive Report for PowerPoint Presentation

---

# 📊 SLIDE 1: PROJECT OVERVIEW

## Samsung PRISM - Punjabi Speech Dataset Analysis

**Project Title:** Audio Quality Assessment & Spoof Detection for Punjabi Speech

**Team Members:**
- Harsh Partap Jain (harshpartapjainsdg@gmail.com)
- Gurkirat Singh
- Ashmit Singh

**Objective:** Develop a comprehensive audio quality assessment and spoof detection system for Punjabi speech recordings

---

# 📊 SLIDE 2: DATASET OVERVIEW

## Dataset Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Bonafide (Genuine)** | 564 files | Original authentic recordings |
| **Spoofed-1** | 670 files | First-level replay attacks |
| **Spoofed-2** | 617 files | Second-level replay attacks |
| **Total Samples** | 1,851 files | Complete dataset |

### Speakers:
- **Female:** F1, F2, F3, F4
- **Male:** M1, M2, M3

### Recording Devices:
- iPhone 14 Pro (IP14p)
- iPhone 15 (IP15)
- iPhone 16 Pro (IP16p)
- MacBook Pro 15" (MBP15)
- MacBook Air 13" (MBA13)

### Recording Distances:
- 0m, 0.5m, 1m, 2m, 4m, 5m

---

# 📊 SLIDE 3: QUALITY METRICS USED

## Audio Quality Assessment Metrics

| Metric | Full Name | Range | Meaning |
|--------|-----------|-------|---------|
| **PESQ** | Perceptual Evaluation of Speech Quality | 1.0 - 4.5 | Higher = Better Quality |
| **STOI** | Short-Time Objective Intelligibility | 0.0 - 1.0 | Higher = More Intelligible |
| **SNR** | Signal-to-Noise Ratio | dB | Higher = Less Noise |

### PESQ Score Interpretation:
| Score | Quality Level |
|-------|---------------|
| 4.0 - 4.5 | Excellent |
| 3.0 - 4.0 | Good |
| 2.0 - 3.0 | Fair |
| 1.0 - 2.0 | Poor |

---

# 📊 SLIDE 4: BONAFIDE QUALITY RESULTS

## Genuine Audio Quality Analysis

### Overall Statistics (382 Comparisons):

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **PESQ** | 2.43 | 1.59 | 1.02 | 4.64 |
| **STOI** | 0.72 | 0.25 | 0.17 | 1.00 |
| **SNR Global** | 32.09 dB | 48.56 dB | -3.98 dB | 100.0 dB |

### Quality by Distance:

| Distance | PESQ (Mean) | Sample Count |
|----------|-------------|--------------|
| 0m (Reference) | 4.64 | 100 |
| 1m | 4.64 | 27 |
| 2m | 1.21 | 72 |
| 4m | 1.27 | 99 |
| 5m | 1.50 | 84 |

**Key Finding:** Quality degrades significantly beyond 1m distance

---

# 📊 SLIDE 5: BONAFIDE VS SPOOFED COMPARISON

## Quality Degradation Analysis

### Mean Quality Scores:

| Dataset | PESQ | STOI | SNR (dB) |
|---------|------|------|----------|
| **Bonafide** | 1.30 | 0.58 | -2.53 |
| **Spoofed-1** | 1.16 | 0.47 | -3.98 |
| **Spoofed-2** | 1.15 | 0.35 | -4.14 |
| **All Spoofed** | 1.16 | 0.41 | -4.06 |

### Quality Degradation (Spoofed vs Bonafide):

| Metric | Degradation |
|--------|-------------|
| **PESQ** | -11.2% |
| **STOI** | -28.2% |
| **SNR** | -1.45 dB worse |

---

# 📊 SLIDE 6: SPEAKER-WISE ANALYSIS

## PESQ Scores by Speaker

| Speaker | Bonafide | Spoofed-1 | Spoofed-2 |
|---------|----------|-----------|-----------|
| **F1** | 1.05 | 1.09 | 1.11 |
| **F2** | 1.31 | 1.17 | 1.11 |
| **F3** | - | 1.12 | 1.10 |
| **F4** | 1.35 | 1.20 | 1.21 |
| **M1** | 1.41 | 1.24 | 1.25 |
| **M2** | - | 1.14 | 1.11 |
| **M3** | 1.48 | - | - |

**Best Quality:** M3 (Bonafide PESQ = 1.48)
**Most Consistent:** M1 across all conditions

---

# 📊 SLIDE 7: SPOOF1 vs SPOOF2 DIRECT COMPARISON

## Direct Audio-to-Audio Comparison (1,758 pairs)

### Mean Quality Metrics:

| Metric | Mean Value | Std Dev |
|--------|------------|---------|
| **PESQ** | 1.46 | 0.25 |
| **STOI** | 0.40 | 0.12 |
| **SNR Global** | -1.01 dB | 2.15 |
| **Correlation** | 0.27 | 0.15 |

**Key Finding:** Second replay (Spoof-2) shows further degradation from Spoof-1

---

# 📊 SLIDE 8: SPOOF DETECTION - BASELINE METHOD

## MFCC + Cosine Similarity Approach

### Method:
1. Extract MFCC features from all audio
2. Compute centroid of bonafide samples
3. Score by cosine similarity to centroid
4. Higher score = More likely genuine

### Results:

| Evaluation | EER (%) | minDCF | Cllr (bits) |
|------------|---------|--------|-------------|
| **Spoofed-1 vs Bonafide** | 31.32% | 1.60 | 2.33 |
| **Spoofed-2 vs Bonafide** | 25.31% | 0.87 | 2.31 |

**Observation:** Spoofed-2 is easier to detect (lower EER)

---

# 📊 SLIDE 9: SPOOF DETECTION - ADVANCED METHOD

## LFCC + CQCC + Logistic Regression

### Method:
1. Extract LFCC (Linear Frequency Cepstral Coefficients)
2. Extract CQCC (Constant-Q Cepstral Coefficients)
3. Combine features (80-dimensional vector)
4. Train Logistic Regression classifier

### Results:

| Evaluation | EER (%) | minDCF | Cllr (bits) | AUC |
|------------|---------|--------|-------------|-----|
| **Spoofed-1 vs Bonafide** | 24.63% | 0.12 | 0.72 | 0.84 |
| **Spoofed-2 vs Bonafide** | 20.35% | 0.13 | 0.62 | 0.90 |

### Classifier Accuracy:
- **Spoofed-1:** Training 81.02%, Test 75.71%
- **Spoofed-2:** Training 90.78%, Test 78.48%

---

# 📊 SLIDE 10: COMPARISON - BASELINE vs ADVANCED

## Spoof Detection Performance Comparison

| Method | Spoofed-1 EER | Spoofed-2 EER | Improvement |
|--------|---------------|---------------|-------------|
| **Baseline MFCC** | 31.32% | 25.31% | - |
| **Advanced LFCC+CQCC** | 24.63% | 20.35% | ~6-7% better |

### Key Improvements:
- **EER Reduction:** 6.69% for Spoofed-1, 4.96% for Spoofed-2
- **minDCF Reduction:** 93% better (1.60 → 0.12)
- **AUC:** 0.84 - 0.90 (Good discrimination)

---

# 📊 SLIDE 11: THRESHOLD ANALYSIS

## Detection Threshold Recommendations

### PESQ-based Detection:

| Threshold | Bonafide Pass | Spoofed Detected | Accuracy |
|-----------|---------------|------------------|----------|
| PESQ > 1.1 | 76.3% | 25.4% | 33.9% |
| PESQ > 1.2 | 74.3% | 76.0% | 75.5% |
| **PESQ > 1.3** | 56.5% | **94.9%** | **88.6%** |
| PESQ > 1.4 | 32.4% | 99.1% | 88.0% |

### STOI-based Detection:

| Threshold | Bonafide Pass | Spoofed Detected | Accuracy |
|-----------|---------------|------------------|----------|
| STOI > 0.5 | 75.9% | 64.1% | 67.0% |
| **STOI > 0.6** | 73.1% | **93.9%** | **90.6%** |
| STOI > 0.7 | 21.3% | 99.6% | 86.7% |

**Best Thresholds:**
- PESQ < 1.3: Detects 94.9% spoofed (misses 43.5% bonafide)
- STOI < 0.6: Detects 93.9% spoofed (misses 26.9% bonafide)

---

# 📊 SLIDE 12: KEY FINDINGS SUMMARY

## Major Discoveries

### 1. Quality Degradation Pattern:
- **Distance Impact:** Quality drops significantly beyond 1m
- **Replay Impact:** Each replay level degrades quality by ~10-15%
- **STOI more sensitive** than PESQ for spoof detection

### 2. Spoof Detection Effectiveness:
- **Best Method:** LFCC+CQCC with Logistic Regression
- **Best EER:** 20.35% (Spoofed-2 detection)
- **Best AUC:** 0.90

### 3. Device Performance:
| Rank | Device | PESQ Score |
|------|--------|------------|
| 1 | iPhone 14 Pro | 1.20 |
| 2 | iPhone 16 Pro | 1.18 |
| 3 | iPhone 16 | 1.16 |

### 4. Statistical Significance:
- Spoofed-1 vs Spoofed-2 difference: **p < 0.000001** (Highly Significant)

---

# 📊 SLIDE 13: TECHNICAL ARCHITECTURE

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDIO INPUT                               │
│         (Bonafide / Spoofed-1 / Spoofed-2)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                              │
│  • Resample to 16kHz  • Mono conversion  • Silence trim     │
│  • RMS normalization (-20 dB)                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │  MFCC    │   │  LFCC    │   │  CQCC    │                 │
│  │ (20 dim) │   │ (40 dim) │   │ (40 dim) │                 │
│  └──────────┘   └──────────┘   └──────────┘                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   QUALITY METRICS                            │
│     PESQ (ITU-T P.862)  │  STOI  │  SNR (Global/Segmental) │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 SPOOF DETECTION                              │
│  ┌─────────────────┐       ┌─────────────────────────────┐  │
│  │ Baseline:       │       │ Advanced:                   │  │
│  │ Cosine Sim to   │       │ LFCC+CQCC + Logistic Reg   │  │
│  │ Bonafide Center │       │ 80-dim → Binary Classifier │  │
│  └─────────────────┘       └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

# 📊 SLIDE 14: EVALUATION METRICS EXPLAINED

## Spoof Detection Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **EER** | Equal Error Rate (FPR = FNR) | Lower is better (0% ideal) |
| **minDCF** | Minimum Detection Cost Function | Lower is better |
| **Cllr** | Log-likelihood Ratio Cost | Lower is better |
| **AUC** | Area Under ROC Curve | Higher is better (1.0 ideal) |

### Our Results Summary:
- **Best EER:** 20.35% (Advanced, Spoofed-2)
- **Best AUC:** 0.90 (Advanced, Spoofed-2)
- **Improvement over baseline:** ~7% EER reduction

---

# 📊 SLIDE 15: CONCLUSIONS & FUTURE WORK

## Conclusions

### ✅ Achievements:
1. Built comprehensive audio quality pipeline for Punjabi speech
2. Achieved **20.35% EER** in spoof detection
3. Identified optimal detection thresholds (PESQ < 1.3, STOI < 0.6)
4. Demonstrated LFCC+CQCC superiority over MFCC baseline

### 📈 Key Numbers to Remember:
| Metric | Value |
|--------|-------|
| Total Samples Analyzed | 1,851 |
| Best Detection EER | 20.35% |
| Best AUC | 0.90 |
| Quality Degradation (Spoofed) | 11-28% |

### 🔮 Future Work:
1. Deep learning approaches (CNN, RNN)
2. Real-time spoof detection system
3. Cross-language transfer learning
4. Larger dataset collection

---

# 📊 SLIDE 16: ACKNOWLEDGMENTS

## Acknowledgments

- **Samsung PRISM Program** for research support and funding
- **ITU-T** for PESQ standard (P.862)
- **librosa** developers for audio processing tools

---

# APPENDIX: RAW DATA TABLES

## Complete Quality Statistics

### Bonafide Quality by Speaker:
| Speaker | PESQ Mean | PESQ Std | Sample Count |
|---------|-----------|----------|--------------|
| F1 | 1.05 | 0.06 | 62 |
| F2 | 1.31 | 0.06 | 64 |
| F3 | 4.64 | 0.00 | 50 |
| F4 | 1.35 | 0.08 | 34 |
| M1 | 2.63 | 1.58 | 72 |
| M2 | 4.64 | 0.00 | 50 |
| M3 | 1.61 | 0.63 | 50 |

### Complete Summary Statistics:

| Metric | Bonafide | Spoofed-1 | Spoofed-2 | All Spoofed |
|--------|----------|-----------|-----------|-------------|
| Count | 253 | 668 | 617 | 1285 |
| PESQ Mean | 1.301 | 1.160 | 1.150 | 1.155 |
| PESQ Std | 0.175 | 0.072 | 0.106 | 0.090 |
| PESQ Min | 1.024 | 1.027 | 1.031 | 1.027 |
| PESQ Max | 1.748 | 1.449 | 2.754 | 2.754 |
| STOI Mean | 0.577 | 0.471 | 0.353 | 0.414 |
| STOI Std | 0.181 | 0.140 | 0.115 | 0.141 |
| SNR Mean | -2.53 dB | -3.98 dB | -4.14 dB | -4.06 dB |
| SNR Std | 0.51 dB | 2.21 dB | 2.27 dB | 2.24 dB |

---

## Quick Reference Card for PPT

### 🎯 Project Numbers at a Glance:

| Item | Value |
|------|-------|
| **Total Audio Files** | 1,851 |
| **Bonafide Samples** | 564 |
| **Spoofed Samples** | 1,287 |
| **Speakers** | 7 (4F + 3M) |
| **Recording Devices** | 5 |
| **Distance Range** | 0m - 5m |
| **Sample Rate** | 16 kHz |
| **Best PESQ (Bonafide)** | 4.64 |
| **Avg PESQ (Spoofed)** | 1.16 |
| **Quality Degradation** | 11-28% |
| **Best EER** | 20.35% |
| **Best AUC** | 0.90 |
| **Recommended PESQ Threshold** | < 1.3 |
| **Recommended STOI Threshold** | < 0.6 |

---

*Report Generated: December 12, 2025*
*Samsung PRISM - Punjabi Speech Quality Assessment*
