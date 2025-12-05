#!/usr/bin/env python3
"""
Samsung PRISM - Comprehensive Audio Quality Analysis
=====================================================

Complete analysis of:
1. Bonafide internal comparison (distance degradation)
2. Spoofed-1 vs Bonafide comparison
3. Spoofed-2 vs Bonafide comparison
4. Spoofed-1 vs Spoofed-2 comparison
5. Statistical analysis and detection thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = Path('samsung_prism_results')
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
bonafide_all = pd.read_csv('results_bonafide/latest_results.csv')
spoofed = pd.read_csv('results_spoofed/spoofed_results.csv')

# Filter out self-comparisons from bonafide
bonafide = bonafide_all[bonafide_all['filename'] != bonafide_all['reference_filename']].copy()

# Split spoofed by type
spoof1 = spoofed[spoofed['spoof_type'] == 'Spoofed-1'].copy()
spoof2 = spoofed[spoofed['spoof_type'] == 'Spoofed-2'].copy()

print(f"Bonafide (real comparisons): {len(bonafide)}")
print(f"Spoofed-1: {len(spoof1)}")
print(f"Spoofed-2: {len(spoof2)}")

# =============================================================================
# 1. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("1. SUMMARY STATISTICS")
print("="*80)

def get_stats(df, name, pesq_col='pesq', stoi_col='stoi', snr_col='snr_global_db'):
    return {
        'Dataset': name,
        'Count': len(df),
        'PESQ_mean': df[pesq_col].mean(),
        'PESQ_std': df[pesq_col].std(),
        'PESQ_min': df[pesq_col].min(),
        'PESQ_max': df[pesq_col].max(),
        'STOI_mean': df[stoi_col].mean(),
        'STOI_std': df[stoi_col].std(),
        'SNR_mean': df[snr_col].mean() if snr_col in df.columns else np.nan,
        'SNR_std': df[snr_col].std() if snr_col in df.columns else np.nan,
    }

summary_stats = pd.DataFrame([
    get_stats(bonafide, 'Bonafide'),
    get_stats(spoof1, 'Spoofed-1'),
    get_stats(spoof2, 'Spoofed-2'),
    get_stats(spoofed, 'All Spoofed'),
])

print("\nOverall Statistics:")
print(summary_stats.round(3).to_string(index=False))
summary_stats.to_csv(output_dir / 'summary_statistics.csv', index=False)

# =============================================================================
# 2. SPOOFED-1 vs SPOOFED-2 DETAILED COMPARISON
# =============================================================================
print("\n" + "="*80)
print("2. SPOOFED-1 vs SPOOFED-2 COMPARISON")
print("="*80)

print("\n2.1 Overall Comparison:")
print("-"*50)
print(f"{'Metric':<20} {'Spoofed-1':>15} {'Spoofed-2':>15} {'Difference':>15}")
print("-"*50)
print(f"{'Count':<20} {len(spoof1):>15} {len(spoof2):>15} {len(spoof1)-len(spoof2):>15}")
print(f"{'PESQ (mean)':<20} {spoof1['pesq'].mean():>15.3f} {spoof2['pesq'].mean():>15.3f} {spoof1['pesq'].mean()-spoof2['pesq'].mean():>15.3f}")
print(f"{'PESQ (std)':<20} {spoof1['pesq'].std():>15.3f} {spoof2['pesq'].std():>15.3f} {'-':>15}")
print(f"{'STOI (mean)':<20} {spoof1['stoi'].mean():>15.3f} {spoof2['stoi'].mean():>15.3f} {spoof1['stoi'].mean()-spoof2['stoi'].mean():>15.3f}")
print(f"{'STOI (std)':<20} {spoof1['stoi'].std():>15.3f} {spoof2['stoi'].std():>15.3f} {'-':>15}")
print(f"{'SNR (mean)':<20} {spoof1['snr_global_db'].mean():>15.2f} {spoof2['snr_global_db'].mean():>15.2f} {spoof1['snr_global_db'].mean()-spoof2['snr_global_db'].mean():>15.2f}")

# Statistical significance test
t_pesq, p_pesq = stats.ttest_ind(spoof1['pesq'].dropna(), spoof2['pesq'].dropna())
t_stoi, p_stoi = stats.ttest_ind(spoof1['stoi'].dropna(), spoof2['stoi'].dropna())

print(f"\n2.2 Statistical Significance (t-test):")
print(f"  PESQ: t={t_pesq:.3f}, p={p_pesq:.6f} {'***' if p_pesq < 0.001 else '**' if p_pesq < 0.01 else '*' if p_pesq < 0.05 else 'ns'}")
print(f"  STOI: t={t_stoi:.3f}, p={p_stoi:.6f} {'***' if p_stoi < 0.001 else '**' if p_stoi < 0.01 else '*' if p_stoi < 0.05 else 'ns'}")

# 2.3 By Speaker
print("\n2.3 By Speaker:")
spoof_speaker = spoofed.pivot_table(
    index='speaker_id', 
    columns='spoof_type', 
    values=['pesq', 'stoi'], 
    aggfunc=['mean', 'count']
)
print(spoof_speaker.round(3).to_string())

# 2.4 By Recording Device
print("\n2.4 By Recording Device:")
spoof_device = spoofed.pivot_table(
    index='recording_device',
    columns='spoof_type',
    values=['pesq', 'stoi'],
    aggfunc='mean'
)
print(spoof_device.round(3).to_string())

# 2.5 By Distance
print("\n2.5 By Distance:")
spoof_dist = spoofed.pivot_table(
    index='distance',
    columns='spoof_type',
    values=['pesq', 'stoi'],
    aggfunc=['mean', 'count']
)
print(spoof_dist.round(3).to_string())

# =============================================================================
# 3. BONAFIDE vs SPOOFED COMPARISON
# =============================================================================
print("\n" + "="*80)
print("3. BONAFIDE vs SPOOFED COMPARISON")
print("="*80)

print("\n3.1 Overall Quality Metrics:")
print("-"*70)
print(f"{'Metric':<15} {'Bonafide':>15} {'Spoofed-1':>15} {'Spoofed-2':>15} {'All Spoofed':>15}")
print("-"*70)
print(f"{'PESQ':<15} {bonafide['pesq'].mean():>15.3f} {spoof1['pesq'].mean():>15.3f} {spoof2['pesq'].mean():>15.3f} {spoofed['pesq'].mean():>15.3f}")
print(f"{'STOI':<15} {bonafide['stoi'].mean():>15.3f} {spoof1['stoi'].mean():>15.3f} {spoof2['stoi'].mean():>15.3f} {spoofed['stoi'].mean():>15.3f}")
print(f"{'SNR (dB)':<15} {bonafide['snr_global_db'].mean():>15.2f} {spoof1['snr_global_db'].mean():>15.2f} {spoof2['snr_global_db'].mean():>15.2f} {spoofed['snr_global_db'].mean():>15.2f}")

# Degradation percentages
print("\n3.2 Quality Degradation (vs Bonafide):")
print(f"  Spoofed-1 PESQ degradation: {(1 - spoof1['pesq'].mean()/bonafide['pesq'].mean())*100:.1f}%")
print(f"  Spoofed-2 PESQ degradation: {(1 - spoof2['pesq'].mean()/bonafide['pesq'].mean())*100:.1f}%")
print(f"  Spoofed-1 STOI degradation: {(1 - spoof1['stoi'].mean()/bonafide['stoi'].mean())*100:.1f}%")
print(f"  Spoofed-2 STOI degradation: {(1 - spoof2['stoi'].mean()/bonafide['stoi'].mean())*100:.1f}%")

# =============================================================================
# 4. DETECTION THRESHOLD ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. SPOOF DETECTION THRESHOLD ANALYSIS")
print("="*80)

# PESQ thresholds
pesq_thresholds = [1.1, 1.2, 1.3, 1.4, 1.5]
print("\n4.1 PESQ Threshold Analysis:")
print("-"*70)
print(f"{'Threshold':<12} {'Bonafide Pass%':>15} {'Spoof1 Pass%':>15} {'Spoof2 Pass%':>15} {'Accuracy':>12}")
print("-"*70)

threshold_results = []
for thresh in pesq_thresholds:
    bon_pass = (bonafide['pesq'] > thresh).mean() * 100
    sp1_pass = (spoof1['pesq'] > thresh).mean() * 100
    sp2_pass = (spoof2['pesq'] > thresh).mean() * 100
    # Accuracy = correctly classify bonafide as pass + spoofed as fail
    accuracy = ((bonafide['pesq'] > thresh).sum() + (spoofed['pesq'] <= thresh).sum()) / (len(bonafide) + len(spoofed)) * 100
    print(f"PESQ > {thresh:<5} {bon_pass:>15.1f} {sp1_pass:>15.1f} {sp2_pass:>15.1f} {accuracy:>12.1f}%")
    threshold_results.append({
        'threshold': f'PESQ > {thresh}',
        'bonafide_pass': bon_pass,
        'spoof1_pass': sp1_pass,
        'spoof2_pass': sp2_pass,
        'accuracy': accuracy
    })

# STOI thresholds
stoi_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n4.2 STOI Threshold Analysis:")
print("-"*70)
print(f"{'Threshold':<12} {'Bonafide Pass%':>15} {'Spoof1 Pass%':>15} {'Spoof2 Pass%':>15} {'Accuracy':>12}")
print("-"*70)

for thresh in stoi_thresholds:
    bon_pass = (bonafide['stoi'] > thresh).mean() * 100
    sp1_pass = (spoof1['stoi'] > thresh).mean() * 100
    sp2_pass = (spoof2['stoi'] > thresh).mean() * 100
    accuracy = ((bonafide['stoi'] > thresh).sum() + (spoofed['stoi'] <= thresh).sum()) / (len(bonafide) + len(spoofed)) * 100
    print(f"STOI > {thresh:<5} {bon_pass:>15.1f} {sp1_pass:>15.1f} {sp2_pass:>15.1f} {accuracy:>12.1f}%")
    threshold_results.append({
        'threshold': f'STOI > {thresh}',
        'bonafide_pass': bon_pass,
        'spoof1_pass': sp1_pass,
        'spoof2_pass': sp2_pass,
        'accuracy': accuracy
    })

pd.DataFrame(threshold_results).to_csv(output_dir / 'threshold_analysis.csv', index=False)

# =============================================================================
# 5. BY SPEAKER ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("5. SPEAKER-WISE ANALYSIS")
print("="*80)

# Get common speakers
speakers = ['F1', 'F2', 'F3', 'F4', 'M1', 'M2', 'M3']

print("\n5.1 PESQ by Speaker:")
print("-"*70)
print(f"{'Speaker':<10} {'Bonafide':>12} {'Spoofed-1':>12} {'Spoofed-2':>12} {'Diff (B-S1)':>12}")
print("-"*70)

speaker_analysis = []
for spk in speakers:
    bon_pesq = bonafide[bonafide['speaker_id'] == spk]['pesq'].mean() if len(bonafide[bonafide['speaker_id'] == spk]) > 0 else np.nan
    sp1_pesq = spoof1[spoof1['speaker_id'] == spk]['pesq'].mean() if len(spoof1[spoof1['speaker_id'] == spk]) > 0 else np.nan
    sp2_pesq = spoof2[spoof2['speaker_id'] == spk]['pesq'].mean() if len(spoof2[spoof2['speaker_id'] == spk]) > 0 else np.nan
    diff = bon_pesq - sp1_pesq if not np.isnan(bon_pesq) and not np.isnan(sp1_pesq) else np.nan
    print(f"{spk:<10} {bon_pesq:>12.3f} {sp1_pesq:>12.3f} {sp2_pesq:>12.3f} {diff:>12.3f}" if not np.isnan(bon_pesq) else f"{spk:<10} {'N/A':>12} {sp1_pesq:>12.3f} {sp2_pesq:>12.3f} {'N/A':>12}")
    speaker_analysis.append({
        'speaker': spk,
        'bonafide_pesq': bon_pesq,
        'spoof1_pesq': sp1_pesq,
        'spoof2_pesq': sp2_pesq
    })

pd.DataFrame(speaker_analysis).to_csv(output_dir / 'speaker_analysis.csv', index=False)

# =============================================================================
# 6. DEVICE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("6. RECORDING DEVICE ANALYSIS")
print("="*80)

device_stats = spoofed.groupby(['spoof_type', 'recording_device']).agg({
    'pesq': ['mean', 'std', 'count'],
    'stoi': ['mean', 'std'],
    'snr_global_db': ['mean', 'std']
}).round(3)

print("\n6.1 Device Performance:")
print(device_stats.to_string())

# Best and worst devices
print("\n6.2 Device Rankings (by PESQ):")
device_ranking = spoofed.groupby('recording_device')['pesq'].mean().sort_values(ascending=False)
print("Best devices:")
for i, (dev, pesq) in enumerate(device_ranking.head(3).items(), 1):
    print(f"  {i}. {dev}: PESQ = {pesq:.3f}")
print("Worst devices:")
for i, (dev, pesq) in enumerate(device_ranking.tail(3).items(), 1):
    print(f"  {i}. {dev}: PESQ = {pesq:.3f}")

# =============================================================================
# 7. GENERATE VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig_size = (12, 8)

# 7.1 PESQ Distribution Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(bonafide['pesq'], bins=30, alpha=0.7, label='Bonafide', color='green')
axes[0].axvline(bonafide['pesq'].mean(), color='darkgreen', linestyle='--', label=f'Mean: {bonafide["pesq"].mean():.2f}')
axes[0].set_xlabel('PESQ Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Bonafide PESQ Distribution')
axes[0].legend()

axes[1].hist(spoof1['pesq'], bins=30, alpha=0.7, label='Spoofed-1', color='orange')
axes[1].axvline(spoof1['pesq'].mean(), color='darkorange', linestyle='--', label=f'Mean: {spoof1["pesq"].mean():.2f}')
axes[1].set_xlabel('PESQ Score')
axes[1].set_title('Spoofed-1 PESQ Distribution')
axes[1].legend()

axes[2].hist(spoof2['pesq'], bins=30, alpha=0.7, label='Spoofed-2', color='red')
axes[2].axvline(spoof2['pesq'].mean(), color='darkred', linestyle='--', label=f'Mean: {spoof2["pesq"].mean():.2f}')
axes[2].set_xlabel('PESQ Score')
axes[2].set_title('Spoofed-2 PESQ Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig(output_dir / 'pesq_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pesq_distribution.png")

# 7.2 Box Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data for box plot
plot_data = pd.DataFrame({
    'PESQ': pd.concat([bonafide['pesq'], spoof1['pesq'], spoof2['pesq']]),
    'Category': ['Bonafide']*len(bonafide) + ['Spoofed-1']*len(spoof1) + ['Spoofed-2']*len(spoof2)
})

plot_data_stoi = pd.DataFrame({
    'STOI': pd.concat([bonafide['stoi'], spoof1['stoi'], spoof2['stoi']]),
    'Category': ['Bonafide']*len(bonafide) + ['Spoofed-1']*len(spoof1) + ['Spoofed-2']*len(spoof2)
})

colors = {'Bonafide': 'green', 'Spoofed-1': 'orange', 'Spoofed-2': 'red'}
sns.boxplot(x='Category', y='PESQ', data=plot_data, ax=axes[0], palette=colors)
axes[0].set_title('PESQ Score Comparison')
axes[0].set_ylabel('PESQ Score')

sns.boxplot(x='Category', y='STOI', data=plot_data_stoi, ax=axes[1], palette=colors)
axes[1].set_title('STOI Score Comparison')
axes[1].set_ylabel('STOI Score')

plt.tight_layout()
plt.savefig(output_dir / 'boxplot_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: boxplot_comparison.png")

# 7.3 Heatmap by Speaker and Spoof Type
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PESQ heatmap
pesq_pivot = spoofed.pivot_table(index='speaker_id', columns='spoof_type', values='pesq', aggfunc='mean')
sns.heatmap(pesq_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0], vmin=1.0, vmax=1.5)
axes[0].set_title('Mean PESQ by Speaker and Spoof Type')

# STOI heatmap
stoi_pivot = spoofed.pivot_table(index='speaker_id', columns='spoof_type', values='stoi', aggfunc='mean')
sns.heatmap(stoi_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1], vmin=0.2, vmax=0.7)
axes[1].set_title('Mean STOI by Speaker and Spoof Type')

plt.tight_layout()
plt.savefig(output_dir / 'heatmap_speaker_spoof.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: heatmap_speaker_spoof.png")

# 7.4 Device Comparison
fig, ax = plt.subplots(figsize=(12, 6))
device_pesq = spoofed.groupby(['recording_device', 'spoof_type'])['pesq'].mean().unstack()
device_pesq.plot(kind='bar', ax=ax, color=['orange', 'red'])
ax.set_xlabel('Recording Device')
ax.set_ylabel('Mean PESQ')
ax.set_title('PESQ by Recording Device and Spoof Type')
ax.legend(title='Spoof Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'device_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: device_comparison.png")

# 7.5 Scatter plot PESQ vs STOI
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(bonafide['pesq'], bonafide['stoi'], alpha=0.5, label='Bonafide', c='green', s=30)
ax.scatter(spoof1['pesq'], spoof1['stoi'], alpha=0.5, label='Spoofed-1', c='orange', s=30)
ax.scatter(spoof2['pesq'], spoof2['stoi'], alpha=0.5, label='Spoofed-2', c='red', s=30)
ax.set_xlabel('PESQ Score')
ax.set_ylabel('STOI Score')
ax.set_title('PESQ vs STOI: Bonafide vs Spoofed')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'pesq_vs_stoi_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pesq_vs_stoi_scatter.png")

# =============================================================================
# 8. EXPORT ALL RESULTS
# =============================================================================
print("\n" + "="*80)
print("8. EXPORTING RESULTS")
print("="*80)

# Export detailed CSVs
bonafide.to_csv(output_dir / 'bonafide_results.csv', index=False)
spoof1.to_csv(output_dir / 'spoofed1_results.csv', index=False)
spoof2.to_csv(output_dir / 'spoofed2_results.csv', index=False)
spoofed.to_csv(output_dir / 'all_spoofed_results.csv', index=False)

print(f"  Saved: bonafide_results.csv ({len(bonafide)} rows)")
print(f"  Saved: spoofed1_results.csv ({len(spoof1)} rows)")
print(f"  Saved: spoofed2_results.csv ({len(spoof2)} rows)")
print(f"  Saved: all_spoofed_results.csv ({len(spoofed)} rows)")

# =============================================================================
# 9. FINAL SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("SAMSUNG PRISM - FINAL SUMMARY REPORT")
print("="*80)

report = f"""
SAMSUNG PRISM PUNJABI SPEECH DATASET ANALYSIS
==============================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET OVERVIEW
-------------------
- Bonafide (real comparisons): {len(bonafide)} samples
- Spoofed-1: {len(spoof1)} samples  
- Spoofed-2: {len(spoof2)} samples
- Total Spoofed: {len(spoofed)} samples

2. QUALITY METRICS SUMMARY
--------------------------
                    Bonafide    Spoofed-1    Spoofed-2
PESQ (mean)         {bonafide['pesq'].mean():.3f}       {spoof1['pesq'].mean():.3f}        {spoof2['pesq'].mean():.3f}
PESQ (std)          {bonafide['pesq'].std():.3f}       {spoof1['pesq'].std():.3f}        {spoof2['pesq'].std():.3f}
STOI (mean)         {bonafide['stoi'].mean():.3f}       {spoof1['stoi'].mean():.3f}        {spoof2['stoi'].mean():.3f}
STOI (std)          {bonafide['stoi'].std():.3f}       {spoof1['stoi'].std():.3f}        {spoof2['stoi'].std():.3f}
SNR (mean dB)       {bonafide['snr_global_db'].mean():.2f}       {spoof1['snr_global_db'].mean():.2f}        {spoof2['snr_global_db'].mean():.2f}

3. KEY FINDINGS
---------------
a) Spoofed-1 vs Spoofed-2:
   - Spoofed-1 has {'higher' if spoof1['pesq'].mean() > spoof2['pesq'].mean() else 'lower'} PESQ ({spoof1['pesq'].mean():.3f} vs {spoof2['pesq'].mean():.3f})
   - Spoofed-1 has {'higher' if spoof1['stoi'].mean() > spoof2['stoi'].mean() else 'lower'} STOI ({spoof1['stoi'].mean():.3f} vs {spoof2['stoi'].mean():.3f})
   - Statistical significance: p = {p_stoi:.6f} {'(significant)' if p_stoi < 0.05 else '(not significant)'}

b) Bonafide vs Spoofed:
   - PESQ degradation: {(1 - spoofed['pesq'].mean()/bonafide['pesq'].mean())*100:.1f}%
   - STOI degradation: {(1 - spoofed['stoi'].mean()/bonafide['stoi'].mean())*100:.1f}%

c) Best Recording Devices:
   1. {device_ranking.index[0]}: PESQ = {device_ranking.iloc[0]:.3f}
   2. {device_ranking.index[1]}: PESQ = {device_ranking.iloc[1]:.3f}
   3. {device_ranking.index[2]}: PESQ = {device_ranking.iloc[2]:.3f}

4. SPOOF DETECTION RECOMMENDATIONS
----------------------------------
- PESQ threshold < 1.3: Detects {(spoofed['pesq'] <= 1.3).mean()*100:.1f}% of spoofed, misses {(bonafide['pesq'] <= 1.3).mean()*100:.1f}% bonafide
- STOI threshold < 0.5: Detects {(spoofed['stoi'] <= 0.5).mean()*100:.1f}% of spoofed, misses {(bonafide['stoi'] <= 0.5).mean()*100:.1f}% bonafide

5. OUTPUT FILES
---------------
- samsung_prism_results/summary_statistics.csv
- samsung_prism_results/threshold_analysis.csv
- samsung_prism_results/speaker_analysis.csv
- samsung_prism_results/bonafide_results.csv
- samsung_prism_results/spoofed1_results.csv
- samsung_prism_results/spoofed2_results.csv
- samsung_prism_results/all_spoofed_results.csv
- samsung_prism_results/pesq_distribution.png
- samsung_prism_results/boxplot_comparison.png
- samsung_prism_results/heatmap_speaker_spoof.png
- samsung_prism_results/device_comparison.png
- samsung_prism_results/pesq_vs_stoi_scatter.png
"""

print(report)

# Save report
with open(output_dir / 'final_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nAll results saved to: {output_dir}/")
print("="*80)
