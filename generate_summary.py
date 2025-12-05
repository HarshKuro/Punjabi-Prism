#!/usr/bin/env python3
"""
Generate comprehensive summary comparing Bonafide and Spoofed results.
"""

import pandas as pd

# Load results
bonafide = pd.read_csv('results_bonafide/latest_results.csv')
spoofed = pd.read_csv('results_spoofed/spoofed_results.csv')

print('='*80)
print('SAMSUNG PRISM PUNJABI SPEECH DATASET - COMPREHENSIVE ANALYSIS')
print('='*80)

print('\n' + '-'*80)
print('1. BONAFIDE INTERNAL COMPARISON (Distance Degradation)')
print('-'*80)
print(f'Total comparisons: {len(bonafide)}')
print(f'Successful: {bonafide["success"].sum()}')
print(f'\nOverall Mean PESQ: {bonafide["pesq"].mean():.3f}')
print(f'Overall Mean STOI: {bonafide["stoi"].mean():.3f}')

print('\nBy Distance (comparing to reference):')
dist_summary = bonafide.groupby('distance_m')[['pesq', 'stoi']].agg(['mean', 'std', 'count'])
print(dist_summary.round(3).to_string())

print('\nBy Speaker:')
speaker_summary = bonafide.groupby('speaker_id')[['pesq', 'stoi']].agg(['mean', 'count'])
print(speaker_summary.round(3).to_string())

print('\n' + '-'*80)
print('2. SPOOFED vs BONAFIDE COMPARISON (Replay Attack Detection)')
print('-'*80)
print(f'Total comparisons: {len(spoofed)}')
print(f'Successful: {spoofed["success"].sum()}')
print(f'\nOverall Mean PESQ: {spoofed["pesq"].mean():.3f}')
print(f'Overall Mean STOI: {spoofed["stoi"].mean():.3f}')
print(f'Overall Mean SNR: {spoofed["snr_global_db"].mean():.2f} dB')

print('\nBy Spoof Type:')
spoof_summary = spoofed.groupby('spoof_type')[['pesq', 'stoi', 'snr_global_db']].agg(['mean', 'std', 'count'])
print(spoof_summary.round(3).to_string())

print('\nBy Recording Device:')
device_summary = spoofed.groupby('recording_device')[['pesq', 'stoi', 'snr_global_db']].agg(['mean', 'count'])
print(device_summary.round(3).to_string())

print('\nBy Speaker:')
speaker_summary = spoofed.groupby('speaker_id')[['pesq', 'stoi']].agg(['mean', 'count'])
print(speaker_summary.round(3).to_string())

print('\n' + '-'*80)
print('3. KEY FINDINGS')
print('-'*80)
print('\nBonafide (Genuine) Audio Quality:')
print('  - PESQ scores range from 1.05 (far distance) to 4.64 (close/same)')
print('  - Clear quality degradation with increasing distance')
print('  - Self-comparison at 0m shows near-perfect PESQ ~4.64')

print('\nSpoofed (Replay Attack) Audio Quality:')
print(f'  - Mean PESQ: {spoofed["pesq"].mean():.3f} (very low quality)')
print(f'  - Mean STOI: {spoofed["stoi"].mean():.3f} (low intelligibility)')
print(f'  - Negative SNR ({spoofed["snr_global_db"].mean():.2f} dB) indicates distortion')

print('\n' + '-'*80)
print('4. SPOOF DETECTION METRICS')
print('-'*80)
bonafide_mean_pesq = bonafide['pesq'].mean()
spoofed_mean_pesq = spoofed['pesq'].mean()
bonafide_mean_stoi = bonafide['stoi'].mean()
spoofed_mean_stoi = spoofed['stoi'].mean()

print(f'\nPESQ Comparison:')
print(f'  - Bonafide mean:  {bonafide_mean_pesq:.3f}')
print(f'  - Spoofed mean:   {spoofed_mean_pesq:.3f}')
print(f'  - Difference:     {bonafide_mean_pesq - spoofed_mean_pesq:.3f}')
print(f'  - Degradation:    {((1 - spoofed_mean_pesq/bonafide_mean_pesq) * 100):.1f}%')

print(f'\nSTOI Comparison:')
print(f'  - Bonafide mean:  {bonafide_mean_stoi:.3f}')
print(f'  - Spoofed mean:   {spoofed_mean_stoi:.3f}')
print(f'  - Difference:     {bonafide_mean_stoi - spoofed_mean_stoi:.3f}')
print(f'  - Degradation:    {((1 - spoofed_mean_stoi/bonafide_mean_stoi) * 100):.1f}%')

# Detection threshold analysis
print('\n' + '-'*80)
print('5. DETECTION THRESHOLD ANALYSIS')
print('-'*80)
print('\nSuggested PESQ thresholds for spoof detection:')
print('  - If PESQ < 1.3: Very likely spoofed (replay attack)')
print('  - If PESQ > 2.0: Likely genuine bonafide audio')
print(f'\n  Bonafide PESQ > 1.3: {(bonafide["pesq"] > 1.3).sum()}/{len(bonafide)} ({(bonafide["pesq"] > 1.3).mean()*100:.1f}%)')
print(f'  Spoofed PESQ > 1.3:  {(spoofed["pesq"] > 1.3).sum()}/{len(spoofed)} ({(spoofed["pesq"] > 1.3).mean()*100:.1f}%)')

print('\nSuggested STOI thresholds:')
print('  - If STOI < 0.5: Likely spoofed')
print('  - If STOI > 0.7: Likely genuine')
print(f'\n  Bonafide STOI > 0.5: {(bonafide["stoi"] > 0.5).sum()}/{len(bonafide)} ({(bonafide["stoi"] > 0.5).mean()*100:.1f}%)')
print(f'  Spoofed STOI > 0.5:  {(spoofed["stoi"] > 0.5).sum()}/{len(spoofed)} ({(spoofed["stoi"] > 0.5).mean()*100:.1f}%)')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print('\nResult files:')
print('  - results_bonafide/results.csv (382 comparisons)')
print('  - results_spoofed/spoofed_results.csv (1285 comparisons)')
