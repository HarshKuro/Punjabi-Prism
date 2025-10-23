# Critical Issues and Limitations Analysis

## Project: Audio Quality Assessment Framework for Punjabi Speech Dataset

---

## 1. FUNDAMENTAL METHODOLOGICAL FLAWS

### 1.1 **FALSE PESQ Implementation in Single-File Estimation**

**Issue:** The synthetic PESQ estimation in `process_all_datasets.py` is **NOT actual PESQ**.

**Problems:**
- PESQ is a **standardized perceptual model** (ITU-T P.862) that requires **two signals** (reference + degraded)
- The synthetic "PESQ" is just a weighted combination of spectral features
- **Misleading naming** - calling it "PESQ" when it's actually a custom quality score
- **No validation** against real PESQ scores to prove correlation
- **Arbitrary coefficients** in the estimation formula with no scientific basis

**Impact:** Research conclusions based on synthetic PESQ are **scientifically invalid**.

### 1.2 **Incorrect SNR Calculation in Single-File Mode**

**Issue:** True SNR requires a clean reference signal. Single-file SNR estimation is **mathematically impossible**.

**Problems:**
- SNR = Signal Power / Noise Power - requires **known clean signal**
- Voice Activity Detection (VAD) approach assumes speech/silence separation equals signal/noise
- **False assumption:** Silence periods contain only noise (often contain room tone, mic noise, etc.)
- **Misleading results:** Negative SNR values don't reflect actual signal-to-noise ratio

**Mathematical Reality:**
```
True SNR = 10 × log₁₀(∑ clean² / ∑ (clean - degraded)²)
Fake SNR = 10 × log₁₀(∑ voiced_frames² / ∑ silence_frames²)
```

### 1.3 **STOI Estimation Without Reference**

**Issue:** STOI (Short-Time Objective Intelligibility) is a **reference-based metric**.

**Problems:**
- Original STOI compares time-frequency representations of clean and degraded signals
- Synthetic STOI based on spectral rolloff is **not STOI at all**
- **No correlation** with actual intelligibility measurements
- **Misleading metric name**

---

## 2. DATASET AND REFERENCE ISSUES

### 2.1 **Questionable Reference Selection**

**Issue:** Using F3 (0m distance) as "clean" references is **problematic**.

**Problems:**
- F3 recordings are **not clean references** - they're recorded in the same environment
- Distance labels may be **incorrect** (0m doesn't mean clean recording)
- **Same acoustic environment** means similar noise characteristics
- **Speaker mismatch** between F3 references and other datasets
- **No validation** that F3 files are actually higher quality

### 2.2 **Distance Extraction Logic Errors**

**Issue:** Distance parsing from folder names contains **logical inconsistencies**.

**Problems:**
```python
# WRONG: F21m, F22m, F24m -> 1m, 2m, 4m
# This assumes F21m means 1m, but folder name suggests 21m
r'F(\d+)(\d)m',     # F21m, F22m, F24m -> 1m, 2m, 4m
```

**Evidence from Results:**
- Script shows: "F21m (Distance: 21m)" but processes as 1m
- **Inconsistent naming convention** causes wrong distance assignments
- **No validation** of extracted distances against actual recording conditions

### 2.3 **Speaker Matching Assumptions**

**Issue:** Assuming F3 speakers match F1, F2, M2 speakers is **unvalidated**.

**Problems:**
- **No proof** that S01 in F3 is the same person as S01 in F1/F2
- **Voice characteristics** may differ significantly between sessions
- **Recording conditions** may vary between datasets
- **Gender mismatch** (using female F3 references for male M2 voices)

---

## 3. STATISTICAL AND SCIENTIFIC ISSUES

### 3.1 **False Statistical Claims**

**Issue:** README contains **unsubstantiated statistical analysis**.

**Problematic Claims:**
```markdown
- Correlation coefficient (PESQ): r = 0.73  # NO VALIDATION SHOWN
- ANOVA Results: F(4,218) = 45.6, p < 0.001  # NO ACTUAL ANOVA PERFORMED
- Distance classification: 94% accuracy  # NO GROUND TRUTH FOR VALIDATION
```

**Problems:**
- **No statistical analysis code** provided
- **No validation datasets** to compute correlations
- **Fabricated results** without computational backing
- **Misleading research claims**

### 3.2 **Regression Analysis Issues**

**Issue:** Distance-quality relationships are **not validated**.

**Problems:**
```markdown
PESQ = 4.26 - 0.68 × log(distance + 1)  (R² = 0.81)  # UNVERIFIED
```
- **No regression analysis code**
- **No scatter plots or residual analysis**
- **R² values without supporting data**
- **Overgeneralized conclusions**

---

## 4. TECHNICAL IMPLEMENTATION PROBLEMS

### 4.1 **Audio Processing Inconsistencies**

**Issue:** Different preprocessing between scripts causes **incomparable results**.

**Problems:**
- `truescore.py`: Normalization by max amplitude
- `process_all_datasets.py`: Different feature extraction methods
- **Inconsistent sampling rate handling**
- **No standardized preprocessing pipeline**

### 4.2 **Error Handling Weaknesses**

**Issue:** Poor error handling leads to **silent failures**.

**Problems:**
```python
except Exception as e:
    print(f"❌ No reference found for {audio_file}")
    continue  # SILENT FAILURE - NO LOGGING
```
- **Missing files ignored** without proper logging
- **No validation** of audio file integrity
- **No checks** for empty or corrupted files
- **Processing continues** with partial data

### 4.3 **Performance and Scalability Issues**

**Issue:** Inefficient processing and **memory problems**.

**Problems:**
- **Loading entire audio files** into memory unnecessarily
- **No parallel processing** for batch operations
- **Redundant feature calculations**
- **No progress tracking** for large datasets
- **Memory leaks** from librosa operations

---

## 5. RESEARCH VALIDITY CONCERNS

### 5.1 **Reproducibility Issues**

**Issue:** Results are **not reproducible** due to multiple factors.

**Problems:**
- **Random elements** in synthetic estimators without fixed seeds
- **Environment-dependent** librosa behavior
- **No version pinning** for critical libraries
- **Undefined processing order** affects results
- **No containerization** for consistent environments

### 5.2 **Validation Gaps**

**Issue:** **No ground truth validation** for any metric.

**Problems:**
- **No human listening tests** to validate quality ratings
- **No comparison** with professional audio analysis tools
- **No cross-validation** with other PESQ implementations
- **No statistical significance testing**
- **No confidence intervals** for reported metrics

### 5.3 **Bias and Assumptions**

**Issue:** Multiple **unacknowledged biases** in the analysis.

**Problems:**
- **Language bias**: Optimized for Punjabi without justification
- **Speaker bias**: Assumes female voices as reference quality
- **Environmental bias**: Assumes controlled recording conditions
- **Cultural bias**: Quality thresholds may not apply universally

---

## 6. DOCUMENTATION AND PRESENTATION ISSUES

### 6.1 **Misleading Research Paper Format**

**Issue:** README formatted as research paper **without peer review or validation**.

**Problems:**
- **False academic credibility** through formatting
- **Unsubstantiated claims** presented as research findings
- **Missing methodology validation**
- **No ethical review** for human subjects (voice recordings)
- **No institutional affiliation** or oversight

### 6.2 **Inflated Performance Claims**

**Issue:** **Unrealistic accuracy claims** without supporting evidence.

**Examples:**
```markdown
- PESQ estimation accuracy: ±0.45 MOS  # NO VALIDATION STUDY
- Distance classification: 94% accuracy  # NO GROUND TRUTH
- Processing Speed: ~15 files/minute  # NO BENCHMARKING
```

### 6.3 **Missing Critical Disclaimers**

**Issue:** **No warnings** about limitations and inappropriate use cases.

**Missing Disclaimers:**
- Synthetic metrics are **not equivalent** to standardized measures
- Results **not validated** against professional tools
- **Not suitable** for commercial or clinical applications
- **Research use only** with significant limitations

---

## 7. ETHICAL AND PROFESSIONAL CONCERNS

### 7.1 **Misrepresentation of Scientific Methods**

**Issue:** Presenting **non-standard methods** as established techniques.

**Problems:**
- **False scientific authority** through technical jargon
- **Misleading metric names** (synthetic PESQ, fake SNR)
- **Unvalidated algorithms** presented as research contributions
- **No peer review** or scientific oversight

### 7.2 **Potential Misuse of Results**

**Issue:** Results could be **misused** for important decisions.

**Risks:**
- **Quality control decisions** based on invalid metrics
- **Research conclusions** built on flawed foundations
- **Commercial applications** using unvalidated quality scores
- **Academic citations** propagating methodological errors

---

## 8. SPECIFIC CODE QUALITY ISSUES

### 8.1 **Magic Numbers and Arbitrary Constants**

**Issue:** Critical parameters **hardcoded without justification**.

**Examples:**
```python
base_pesq = 2.5  # WHY 2.5? NO JUSTIFICATION
centroid_factor = np.clip(np.mean(spectral_centroid) / 2000, 0.5, 2.0)  # WHY 2000?
threshold = np.percentile(energy, 30)  # WHY 30th PERCENTILE?
```

### 8.2 **Inconsistent Data Types and Formats**

**Issue:** **Type inconsistencies** cause downstream problems.

**Problems:**
```python
distance = int(distance_match.group(1))  # Sometimes int
distance = 0  # Sometimes literal
np.int64(1)  # NumPy integers in results
```

### 8.3 **Poor Function Design**

**Issue:** Functions with **unclear purposes and side effects**.

**Problems:**
- **Mixed responsibilities** (processing + formatting + saving)
- **Global state dependencies**
- **Unclear return types**
- **No input validation**

---

## 9. MISSING CRITICAL FEATURES

### 9.1 **No Quality Control**

**Issue:** **No validation** of input data quality.

**Missing Features:**
- Audio file integrity checks
- Sampling rate validation
- Dynamic range analysis
- Clipping detection
- Silence detection

### 9.2 **No Comparative Analysis**

**Issue:** **No comparison** with established tools.

**Missing Comparisons:**
- MATLAB Audio Toolbox PESQ
- ITU-T reference implementation
- Professional audio analysis software
- Other research implementations

### 9.3 **No Uncertainty Quantification**

**Issue:** **No confidence intervals** or uncertainty measures.

**Missing Elements:**
- Bootstrap confidence intervals
- Cross-validation error estimates
- Sensitivity analysis
- Robustness testing

---

## 10. RECOMMENDATIONS FOR FIXING

### 10.1 **Immediate Actions Required**

1. **Rename synthetic metrics** to avoid confusion (e.g., "Custom Quality Score" instead of "PESQ")
2. **Add clear disclaimers** about limitations and non-standard methods
3. **Remove false statistical claims** from documentation
4. **Validate distance extraction** against actual recording metadata
5. **Fix reference matching logic** or acknowledge mismatches

### 10.2 **Long-term Improvements Needed**

1. **Conduct proper validation study** with ground truth data
2. **Implement standardized preprocessing** pipeline
3. **Add professional quality control** measures
4. **Perform statistical analysis** with proper methodology
5. **Seek peer review** before making research claims

### 10.3 **Ethical Considerations**

1. **Acknowledge limitations** prominently in all documentation
2. **Clarify intended use cases** (educational/experimental only)
3. **Warn against** commercial or clinical applications
4. **Provide contact information** for questions and corrections
5. **Consider retracting** inflated accuracy claims

---

## CONCLUSION

This project contains **fundamental methodological flaws** that render many of its results **scientifically invalid**. The most serious issues are:

1. **Misnamed metrics** that don't correspond to established standards
2. **Unvalidated synthetic algorithms** presented as established methods  
3. **False statistical claims** without supporting analysis
4. **Questionable reference selection** undermining comparative results
5. **Missing validation** against ground truth or professional tools

**Recommendation: This project should be treated as an experimental prototype only, not as a reliable research tool or basis for scientific conclusions.**

---

**Document Version:** 1.0  
**Last Updated:** October 2025  
**Severity Level:** HIGH - Multiple critical issues affecting scientific validity
