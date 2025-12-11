# spoof_detection_metrics.py
"""
Punjabi Speech Spoof Detection Evaluation
==========================================
Samsung PRISM - Punjabi Speech Dataset

This module provides comprehensive evaluation metrics for spoof detection:
- MFCC-based baseline with cosine similarity scoring
- LFCC + CQCC feature extraction with Logistic Regression classifier
- EER (Equal Error Rate)
- minDCF (Minimum Detection Cost Function)
- actDCF (Actual Detection Cost Function)
- Cllr (Log-likelihood Ratio Cost)

Authors: Harsh Partap Jain, Gurkirat Singh, Ashmit Singh
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import librosa
import scipy.fftpack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION - Punjabi Dataset Paths
# ==============================================================================
# Update these paths to your Punjabi dataset location
PATH_BONAFIDE = r"C:\Users\Harsh Jain\Downloads\prism\Bonafide"
# Spoofed data is nested: Spoofed/{speaker}/Spoofed-1/ and Spoofed-2/
PATH_SPOOFED = r"C:\Users\Harsh Jain\Downloads\prism\Spoofed"

# Audio parameters
SAMPLE_RATE = 16000
MIN_DURATION = 0.2  # Skip files shorter than this (seconds)

# Feature extraction parameters
N_MFCC = 20
N_LFCC = 20
N_CQCC = 20
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FILTERS = 40

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def list_audio_files(folder: str) -> List[str]:
    """Recursively find all audio files in a folder."""
    extensions = ['wav', 'flac', 'mp3', 'ogg', 'm4a']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, '**', f'*.{ext}'), recursive=True))
    return sorted(files)


def list_spoofed_files(base_spoofed_path: str, spoof_type: str = "Spoofed-1") -> List[str]:
    """
    List spoofed audio files from the nested structure.
    Structure: Spoofed/{speaker}/Spoofed-1/ or Spoofed-2/
    
    Args:
        base_spoofed_path: Path to the Spoofed folder
        spoof_type: "Spoofed-1" or "Spoofed-2"
    """
    files = []
    extensions = ['wav', 'flac', 'mp3', 'ogg', 'm4a']
    
    # Iterate through speaker folders (f1, f2, m1, etc.)
    if os.path.exists(base_spoofed_path):
        for speaker in os.listdir(base_spoofed_path):
            speaker_path = os.path.join(base_spoofed_path, speaker, spoof_type)
            if os.path.isdir(speaker_path):
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(speaker_path, '**', f'*.{ext}'), recursive=True))
    
    return sorted(files)


def parse_punjabi_filename(filepath: str) -> Dict:
    """
    Parse Punjabi dataset filename convention.
    Format: pa_S{sentence}_{speaker}_{gender}_{device}_{condition}_{distance}_{angle}_{direction}_{noise}_{channel}_{mic}.wav
    Example: pa_S01_f1_female_IP14p_na_1m_90_east_57db_0_B.wav
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    info = {
        'filename': filename,
        'filepath': filepath,
        'language': parts[0] if len(parts) > 0 else 'unknown',
        'sentence_id': parts[1] if len(parts) > 1 else 'unknown',
        'speaker_id': parts[2] if len(parts) > 2 else 'unknown',
        'gender': parts[3] if len(parts) > 3 else 'unknown',
        'device': parts[4] if len(parts) > 4 else 'unknown',
    }
    
    # Try to extract distance
    for part in parts:
        if part.endswith('m') and part[:-1].replace('.', '').isdigit():
            info['distance'] = part
            break
    
    return info


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_mfcc_mean(filepath: str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> Optional[np.ndarray]:
    """Extract mean MFCC features from audio file."""
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        if len(y) < sr * MIN_DURATION:
            print(f"[WARN] File too short: {filepath}")
            return None
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # L2 normalize
        norm = np.linalg.norm(mfcc_mean)
        if norm > 0:
            mfcc_mean = mfcc_mean / norm
        
        return mfcc_mean
    except Exception as e:
        print(f"[WARN] Failed to read {filepath}: {e}")
        return None


def linear_filterbank(n_fft: int, sr: int, n_filters: int = N_FILTERS, 
                      fmin: float = 0, fmax: float = None) -> np.ndarray:
    """Create linear-spaced filterbank for LFCC."""
    if fmax is None:
        fmax = sr / 2
    
    freqs = np.linspace(fmin, fmax, n_filters + 2)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i-1], bins[i], bins[i+1]
        if c > l:
            fb[i-1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i-1, c:r] = (r - np.arange(c, r)) / (r - c)
    
    return fb


def extract_lfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH,
                 n_filters: int = N_FILTERS, n_ceps: int = N_LFCC) -> np.ndarray:
    """Extract LFCC (Linear Frequency Cepstral Coefficients) features."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) ** 2
    fb = linear_filterbank(n_fft, sr, n_filters=n_filters)
    
    feat = np.dot(fb, S[:n_fft // 2 + 1, :])
    feat[feat == 0] = 1e-8
    log_feat = np.log(feat)
    
    ceps = scipy.fftpack.dct(log_feat, axis=0, norm='ortho')[:n_ceps, :]
    ceps = (ceps - ceps.mean(axis=1, keepdims=True)) / (ceps.std(axis=1, keepdims=True) + 1e-9)
    
    feature_vector = np.hstack([ceps.mean(axis=1), ceps.std(axis=1)])
    return feature_vector


def extract_cqcc(y: np.ndarray, sr: int = SAMPLE_RATE, bins_per_octave: int = 24,
                 n_octaves: int = 7, n_ceps: int = N_CQCC) -> np.ndarray:
    """Extract CQCC (Constant-Q Cepstral Coefficients) features."""
    fmin = 20.0
    n_bins = n_octaves * bins_per_octave
    
    C = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=fmin, 
                    n_bins=n_bins, bins_per_octave=bins_per_octave)
    C_mag = np.abs(C)
    C_mag[C_mag == 0] = 1e-8
    
    logC = np.log(C_mag)
    ceps = scipy.fftpack.dct(logC, axis=0, norm='ortho')[:n_ceps, :]
    ceps = (ceps - ceps.mean(axis=1, keepdims=True)) / (ceps.std(axis=1, keepdims=True) + 1e-9)
    
    feature_vector = np.hstack([ceps.mean(axis=1), ceps.std(axis=1)])
    return feature_vector


def extract_combined_features(filepath: str, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Extract combined LFCC + CQCC features from audio file."""
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        if len(y) < sr * MIN_DURATION:
            return None
        
        lfcc_feat = extract_lfcc(y, sr=sr)
        cqcc_feat = extract_cqcc(y, sr=sr)
        
        return np.hstack([lfcc_feat, cqcc_feat])
    except Exception as e:
        print(f"[WARN] Failed to extract features from {filepath}: {e}")
        return None


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def compute_eer(labels: np.ndarray, scores: np.ndarray, 
                pos_label: int = 1) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Equal Error Rate (EER).
    
    Returns:
        eer: Equal Error Rate (0-1)
        threshold: Threshold at EER
        fpr: False Positive Rates
        tpr: True Positive Rates
        thresholds: All thresholds
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    fnr = 1 - tpr
    
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    
    return eer, threshold, fpr, tpr, thresholds


def compute_min_dcf(labels: np.ndarray, scores: np.ndarray, 
                    beta: float = 1.9, C_miss: float = 1.0, C_fa: float = 10.0,
                    pos_label: int = 1) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Minimum Detection Cost Function (minDCF).
    
    Returns:
        min_dcf: Minimum DCF value
        threshold: Threshold at minDCF
        dcf_vals: All DCF values
        thresholds: All thresholds
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    fnr = 1 - tpr
    
    dcf_vals = beta * fnr * C_miss + fpr * C_fa
    idx = np.argmin(dcf_vals)
    
    return dcf_vals[idx], thresholds[idx], dcf_vals, thresholds


def compute_act_dcf(labels: np.ndarray, scores: np.ndarray, threshold: float,
                    beta: float = 1.9, C_miss: float = 1.0, 
                    C_fa: float = 10.0) -> Tuple[float, float, float]:
    """
    Compute Actual Detection Cost Function (actDCF) at a given threshold.
    
    Returns:
        act_dcf: Actual DCF value
        P_miss: Miss probability
        P_fa: False alarm probability
    """
    preds = (scores >= threshold).astype(int)
    
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    
    if pos_count == 0 or neg_count == 0:
        return float('nan'), None, None
    
    P_miss = np.sum((labels == 1) & (preds == 0)) / pos_count
    P_fa = np.sum((labels == 0) & (preds == 1)) / neg_count
    
    act_dcf = beta * P_miss * C_miss + P_fa * C_fa
    
    return act_dcf, P_miss, P_fa


def compute_cllr(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Log-Likelihood Ratio Cost (Cllr) in bits.
    
    Note: Treats scores as LLR-like for baseline; this is a rough indicator.
    """
    s = np.array(scores)
    pos = s[labels == 1]
    neg = s[labels == 0]
    
    eps = 1e-300
    
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    
    def log_cost_pos(x):
        return np.log(1 + 1 / (np.exp(x) + eps))
    
    def log_cost_neg(x):
        return np.log(1 + np.exp(x))
    
    cllr = np.mean(log_cost_pos(pos)) + np.mean(log_cost_neg(neg))
    cllr_bits = cllr / math.log(2)
    
    return cllr_bits


def compute_cllr_proba(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Cllr from probability scores (0-1 range)."""
    eps = 1e-15
    scores = np.clip(scores, eps, 1 - eps)
    llr = np.log(scores / (1 - scores))
    cllr = np.mean(np.log2(1 + np.exp(-llr * (2 * labels - 1))))
    return cllr


# ==============================================================================
# BASELINE EVALUATOR - MFCC + Cosine Similarity
# ==============================================================================

class BaselineMFCCEvaluator:
    """
    Baseline spoof detection using MFCC features and cosine similarity.
    
    Approach:
    - Extract mean MFCC vectors from all audio files
    - Compute centroid of bonafide samples
    - Score each sample by cosine similarity to bonafide centroid
    - Higher score = more likely genuine
    """
    
    def __init__(self, path_bonafide: str, spoof_files: List[str], output_dir: str = None):
        self.path_bonafide = path_bonafide
        self.spoof_files = spoof_files  # Now accepts a list of files directly
        self.output_dir = output_dir or "punjabi_evaluation/results_baseline"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.records = []
        self.centroid = None
        self.scores_df = None
        self.report = None
    
    def load_and_extract_features(self):
        """Load audio files and extract MFCC features."""
        print("=" * 60)
        print("BASELINE MFCC EVALUATOR - Punjabi Dataset")
        print("=" * 60)
        
        bona_files = list_audio_files(self.path_bonafide)
        spoof_files = self.spoof_files  # Use pre-collected spoof files
        
        print(f"\nFound {len(bona_files)} bonafide files")
        print(f"Found {len(spoof_files)} spoofed files")
        
        if len(bona_files) == 0:
            raise ValueError(f"No audio files found in bonafide path: {self.path_bonafide}")
        if len(spoof_files) == 0:
            raise ValueError("No spoofed files provided!")
        
        # Extract features - Bonafide (label=1)
        print("\nExtracting features from bonafide files...")
        for p in bona_files:
            feats = extract_mfcc_mean(p)
            if feats is not None:
                self.records.append({'path': p, 'label': 1, 'feat': feats})
        
        # Extract features - Spoofed (label=0)
        print("Extracting features from spoofed files...")
        for p in spoof_files:
            feats = extract_mfcc_mean(p)
            if feats is not None:
                self.records.append({'path': p, 'label': 0, 'feat': feats})
        
        print(f"\nSuccessfully extracted features from {len(self.records)} files")
        
        if len(self.records) == 0:
            raise ValueError("No valid audio features extracted!")
        
        return self
    
    def compute_centroid(self):
        """Compute bonafide centroid from positive samples."""
        feats_pos = np.stack([r['feat'] for r in self.records if r['label'] == 1])
        self.centroid = np.mean(feats_pos, axis=0)
        
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
        
        print(f"Computed bonafide centroid from {len(feats_pos)} samples")
        return self
    
    def score_trials(self):
        """Score each trial by cosine similarity to centroid."""
        trial_rows = []
        
        for r in self.records:
            score = float(cosine_similarity(
                r['feat'].reshape(1, -1), 
                self.centroid.reshape(1, -1)
            )[0, 0])
            
            trial_id = os.path.splitext(os.path.basename(r['path']))[0]
            file_info = parse_punjabi_filename(r['path'])
            
            trial_rows.append({
                'trial_id': trial_id,
                'score': score,
                'label': int(r['label']),
                'path': r['path'],
                **{k: v for k, v in file_info.items() if k not in ['filepath', 'filename']}
            })
        
        self.scores_df = pd.DataFrame(trial_rows)
        
        scores_csv = os.path.join(self.output_dir, "scores_baseline.csv")
        self.scores_df.to_csv(scores_csv, index=False)
        print(f"\nSaved scores to {scores_csv}")
        
        return self
    
    def evaluate(self):
        """Compute all evaluation metrics."""
        labels = self.scores_df['label'].values
        scores = self.scores_df['score'].values
        
        # EER
        eer, eer_thr, fpr, tpr, thresholds = compute_eer(labels, scores)
        
        # minDCF
        min_dcf, min_dcf_thr, _, _ = compute_min_dcf(labels, scores)
        
        # actDCF at EER threshold
        act_dcf, P_miss, P_fa = compute_act_dcf(labels, scores, threshold=eer_thr)
        
        # Cllr
        cllr = compute_cllr(labels, scores)
        
        self.report = {
            'EER': eer,
            'EER_percent': eer * 100,
            'EER_threshold': eer_thr,
            'minDCF': min_dcf,
            'minDCF_threshold': min_dcf_thr,
            'actDCF_at_EERthreshold': act_dcf,
            'P_miss_at_EERthr': P_miss,
            'P_fa_at_EERthr': P_fa,
            'Cllr_bits': cllr,
            'num_trials': len(self.scores_df),
            'num_bonafide': int(np.sum(labels == 1)),
            'num_spoofed': int(np.sum(labels == 0)),
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Save report
        report_df = pd.DataFrame({k: [v] for k, v in self.report.items() 
                                  if not isinstance(v, np.ndarray)})
        report_csv = os.path.join(self.output_dir, "eval_report_baseline.csv")
        report_df.to_csv(report_csv, index=False)
        
        print("\n" + "=" * 60)
        print("EVALUATION REPORT - Baseline MFCC")
        print("=" * 60)
        print(f"EER: {self.report['EER_percent']:.2f}%")
        print(f"EER Threshold: {self.report['EER_threshold']:.4f}")
        print(f"minDCF: {self.report['minDCF']:.4f}")
        print(f"actDCF (at EER threshold): {self.report['actDCF_at_EERthreshold']:.4f}")
        print(f"Cllr (bits): {self.report['Cllr_bits']:.4f}")
        print(f"Total trials: {self.report['num_trials']}")
        print(f"  - Bonafide: {self.report['num_bonafide']}")
        print(f"  - Spoofed: {self.report['num_spoofed']}")
        
        return self
    
    def plot_roc(self):
        """Plot and save ROC curve."""
        plt.figure(figsize=(8, 8))
        plt.plot(self.report['fpr'], self.report['tpr'], 'b-', linewidth=2, 
                 label=f'ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.scatter([self.report['EER']], [1 - self.report['EER']], 
                    marker='x', color='red', s=100, linewidths=3,
                    label=f"EER = {self.report['EER_percent']:.2f}%")
        
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Baseline MFCC (Punjabi Dataset)", fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        roc_path = os.path.join(self.output_dir, "roc_curve_baseline.png")
        plt.savefig(roc_path, dpi=150)
        print(f"\nSaved ROC curve to {roc_path}")
        plt.close()
        
        return self
    
    def run(self):
        """Run the complete baseline evaluation pipeline."""
        return (self
                .load_and_extract_features()
                .compute_centroid()
                .score_trials()
                .evaluate()
                .plot_roc())


# ==============================================================================
# ADVANCED EVALUATOR - LFCC + CQCC + Logistic Regression
# ==============================================================================

class AdvancedLFCCCQCCEvaluator:
    """
    Advanced spoof detection using LFCC + CQCC features with Logistic Regression.
    
    Approach:
    - Extract combined LFCC and CQCC features
    - Train Logistic Regression classifier
    - Evaluate with train/test split
    """
    
    def __init__(self, path_bonafide: str, spoof_files: List[str], 
                 output_dir: str = None, test_size: float = 0.2):
        self.path_bonafide = path_bonafide
        self.spoof_files = spoof_files  # Now accepts a list of files directly
        self.output_dir = output_dir or "punjabi_evaluation/results_advanced"
        self.test_size = test_size
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.classifier = None
        self.scores = None
        self.report = None
    
    def load_and_extract_features(self):
        """Load audio files and extract LFCC + CQCC features."""
        print("=" * 60)
        print("ADVANCED LFCC+CQCC EVALUATOR - Punjabi Dataset")
        print("=" * 60)
        
        def load_features_from_files(files: List[str], label: int):
            X, y = [], []
            for f in files:
                feat = extract_combined_features(f)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
            return np.array(X) if X else np.array([]).reshape(0, 0), np.array(y)
        
        bona_files = list_audio_files(self.path_bonafide)
        
        print("\nExtracting LFCC+CQCC features from bonafide files...")
        X_bona, y_bona = load_features_from_files(bona_files, 0)  # Bonafide = 0
        print(f"  Extracted {len(X_bona)} bonafide samples")
        
        print("Extracting LFCC+CQCC features from spoofed files...")
        X_spoof, y_spoof = load_features_from_files(self.spoof_files, 1)  # Spoofed = 1
        print(f"  Extracted {len(X_spoof)} spoofed samples")
        
        self.X = np.vstack([X_bona, X_spoof])
        self.y = np.hstack([y_bona, y_spoof])
        
        print(f"\nTotal samples: {len(self.y)}")
        print(f"Feature dimension: {self.X.shape[1]}")
        
        return self
    
    def split_and_scale(self):
        """Split data and apply standard scaling."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            stratify=self.y, 
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTrain set: {len(self.y_train)} samples")
        print(f"Test set: {len(self.y_test)} samples")
        
        return self
    
    def train_classifier(self):
        """Train Logistic Regression classifier."""
        print("\nTraining Logistic Regression classifier...")
        
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)
        
        # Get prediction probabilities
        self.scores = self.classifier.predict_proba(self.X_test)[:, 1]
        
        train_acc = self.classifier.score(self.X_train, self.y_train)
        test_acc = self.classifier.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_acc * 100:.2f}%")
        print(f"Test accuracy: {test_acc * 100:.2f}%")
        
        return self
    
    def evaluate(self):
        """Compute evaluation metrics."""
        # EER using brentq interpolation
        fpr, tpr, thresholds = roc_curve(self.y_test, self.scores)
        fnr = 1 - tpr
        
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer_thresh = float(interp1d(fpr, thresholds)(eer))
        except:
            # Fallback to standard EER computation
            abs_diffs = np.abs(fpr - fnr)
            idx = np.nanargmin(abs_diffs)
            eer = (fpr[idx] + fnr[idx]) / 2.0
            eer_thresh = thresholds[idx]
        
        # minDCF
        def min_dcf(scores, labels, C_miss=1, C_fa=10, pi_spoof=0.05):
            thresholds_sorted = np.sort(scores)
            min_cost = np.inf
            best_thresh = None
            
            for t in thresholds_sorted:
                P_miss = np.mean((scores[labels == 0] < t).astype(float))
                P_fa = np.mean((scores[labels == 1] >= t).astype(float))
                cost = C_miss * pi_spoof * P_miss + C_fa * (1 - pi_spoof) * P_fa
                
                if cost < min_cost:
                    min_cost = cost
                    best_thresh = t
            
            return min_cost, best_thresh
        
        min_dcf_val, min_dcf_thresh = min_dcf(self.scores, self.y_test)
        
        # Cllr
        cllr = compute_cllr_proba(self.scores, self.y_test)
        
        # AUC
        auc = roc_auc_score(self.y_test, self.scores)
        
        self.report = {
            'EER': eer,
            'EER_percent': eer * 100,
            'EER_threshold': eer_thresh,
            'minDCF': min_dcf_val,
            'minDCF_threshold': min_dcf_thresh,
            'Cllr_bits': cllr,
            'AUC': auc,
            'num_test_samples': len(self.y_test),
            'num_bonafide_test': int(np.sum(self.y_test == 0)),
            'num_spoofed_test': int(np.sum(self.y_test == 1)),
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Save report
        report_df = pd.DataFrame({k: [v] for k, v in self.report.items() 
                                  if not isinstance(v, np.ndarray)})
        report_csv = os.path.join(self.output_dir, "eval_report_advanced.csv")
        report_df.to_csv(report_csv, index=False)
        
        # Save scores
        scores_df = pd.DataFrame({
            'score': self.scores,
            'label': self.y_test
        })
        scores_csv = os.path.join(self.output_dir, "scores_advanced.csv")
        scores_df.to_csv(scores_csv, index=False)
        
        print("\n" + "=" * 60)
        print("EVALUATION REPORT - Advanced LFCC+CQCC")
        print("=" * 60)
        print(f"EER: {self.report['EER_percent']:.2f}%")
        print(f"EER Threshold: {self.report['EER_threshold']:.4f}")
        print(f"minDCF: {self.report['minDCF']:.4f}")
        print(f"Cllr (bits): {self.report['Cllr_bits']:.4f}")
        print(f"AUC: {self.report['AUC']:.4f}")
        
        return self
    
    def plot_results(self):
        """Plot ROC curve and confusion matrix."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        ax1 = axes[0]
        ax1.plot(self.report['fpr'], self.report['tpr'], 'b-', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax1.scatter([self.report['EER']], [1 - self.report['EER']], 
                    marker='x', color='red', s=100, linewidths=3,
                    label=f"EER = {self.report['EER_percent']:.2f}%")
        ax1.set_xlabel("False Positive Rate", fontsize=12)
        ax1.set_ylabel("True Positive Rate", fontsize=12)
        ax1.set_title(f"ROC Curve (AUC = {self.report['AUC']:.4f})", fontsize=14)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax2 = axes[1]
        preds = (self.scores >= self.report['EER_threshold']).astype(int)
        cm = confusion_matrix(self.y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Bonafide', 'Spoofed'],
                    yticklabels=['Bonafide', 'Spoofed'])
        ax2.set_xlabel("Predicted", fontsize=12)
        ax2.set_ylabel("Actual", fontsize=12)
        ax2.set_title("Confusion Matrix (at EER threshold)", fontsize=14)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "evaluation_results_advanced.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved evaluation plots to {plot_path}")
        plt.close()
        
        return self
    
    def run(self):
        """Run the complete advanced evaluation pipeline."""
        return (self
                .load_and_extract_features()
                .split_and_scale()
                .train_classifier()
                .evaluate()
                .plot_results())


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_baseline_evaluation(path_bonafide: str = PATH_BONAFIDE, 
                            spoof_type: str = "Spoofed-1"):
    """Run baseline MFCC evaluation."""
    spoof_files = list_spoofed_files(PATH_SPOOFED, spoof_type)
    evaluator = BaselineMFCCEvaluator(path_bonafide, spoof_files)
    evaluator.run()
    return evaluator


def run_advanced_evaluation(path_bonafide: str = PATH_BONAFIDE,
                           spoof_type: str = "Spoofed-1"):
    """Run advanced LFCC+CQCC evaluation."""
    spoof_files = list_spoofed_files(PATH_SPOOFED, spoof_type)
    evaluator = AdvancedLFCCCQCCEvaluator(path_bonafide, spoof_files)
    evaluator.run()
    return evaluator


def run_full_evaluation():
    """Run both baseline and advanced evaluations for all spoof types."""
    print("\n" + "=" * 80)
    print("PUNJABI SPEECH SPOOF DETECTION - FULL EVALUATION")
    print("Samsung PRISM Project")
    print("=" * 80)
    
    # Get spoofed files for both types
    spoof1_files = list_spoofed_files(PATH_SPOOFED, "Spoofed-1")
    spoof2_files = list_spoofed_files(PATH_SPOOFED, "Spoofed-2")
    
    print(f"\nDataset Summary:")
    print(f"  Bonafide path: {PATH_BONAFIDE}")
    print(f"  Spoofed path: {PATH_SPOOFED}")
    print(f"  Spoofed-1 files: {len(spoof1_files)}")
    print(f"  Spoofed-2 files: {len(spoof2_files)}")
    
    results = {}
    
    # Evaluate Spoofed-1
    print("\n\n>>> EVALUATING SPOOFED-1 vs BONAFIDE <<<\n")
    
    print("\n--- Baseline MFCC Evaluation ---")
    baseline_s1 = BaselineMFCCEvaluator(
        PATH_BONAFIDE, spoof1_files,
        output_dir="punjabi_evaluation/results_spoofed1_baseline"
    )
    baseline_s1.run()
    results['spoofed1_baseline'] = baseline_s1.report
    
    print("\n--- Advanced LFCC+CQCC Evaluation ---")
    advanced_s1 = AdvancedLFCCCQCCEvaluator(
        PATH_BONAFIDE, spoof1_files,
        output_dir="punjabi_evaluation/results_spoofed1_advanced"
    )
    advanced_s1.run()
    results['spoofed1_advanced'] = advanced_s1.report
    
    # Evaluate Spoofed-2
    print("\n\n>>> EVALUATING SPOOFED-2 vs BONAFIDE <<<\n")
    
    print("\n--- Baseline MFCC Evaluation ---")
    baseline_s2 = BaselineMFCCEvaluator(
        PATH_BONAFIDE, spoof2_files,
        output_dir="punjabi_evaluation/results_spoofed2_baseline"
    )
    baseline_s2.run()
    results['spoofed2_baseline'] = baseline_s2.report
    
    print("\n--- Advanced LFCC+CQCC Evaluation ---")
    advanced_s2 = AdvancedLFCCCQCCEvaluator(
        PATH_BONAFIDE, spoof2_files,
        output_dir="punjabi_evaluation/results_spoofed2_advanced"
    )
    advanced_s2.run()
    results['spoofed2_advanced'] = advanced_s2.report
    
    # Summary comparison
    print("\n\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    summary_data = []
    for key, report in results.items():
        summary_data.append({
            'Evaluation': key,
            'EER (%)': report['EER_percent'] if 'EER_percent' in report else report['EER'] * 100,
            'minDCF': report['minDCF'],
            'Cllr (bits)': report['Cllr_bits']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    summary_df.to_csv("punjabi_evaluation/summary_comparison.csv", index=False)
    print("\n\nSaved summary to punjabi_evaluation/summary_comparison.csv")
    
    return results


if __name__ == "__main__":
    # Run full evaluation
    run_full_evaluation()
