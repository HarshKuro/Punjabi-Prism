#!/usr/bin/env python3
"""Analyze bonafide results excluding self-comparisons."""

import pandas as pd

df = pd.read_csv('results_bonafide/latest_results.csv')

# Separate self vs actual comparisons
self_comp = df[df['filename'] == df['reference_filename']]
real_comp = df[df['filename'] != df['reference_filename']]

print('='*70)
print('BONAFIDE ANALYSIS - EXCLUDING SELF-COMPARISONS')
print('='*70)

print(f'\nTotal comparisons: {len(df)}')
print(f'Self-comparisons (excluded): {len(self_comp)}')
print(f'Real comparisons: {len(real_comp)}')

print('\n' + '-'*70)
print('REAL COMPARISONS ONLY (different distances):')
print('-'*70)

print(f'\nMean PESQ: {real_comp["pesq"].mean():.3f}')
print(f'Mean STOI: {real_comp["stoi"].mean():.3f}')
print(f'Mean SNR: {real_comp["snr_global_db"].mean():.2f} dB')

print('\nBy Distance (test file distance):')
dist = real_comp.groupby('distance_m')[['pesq', 'stoi', 'snr_global_db']].agg(['mean', 'count'])
print(dist.round(3).to_string())

print('\nBy Speaker:')
spk = real_comp.groupby('speaker_id')[['pesq', 'stoi']].agg(['mean', 'count'])
print(spk.round(3).to_string())

print('\n' + '-'*70)
print('SELF-COMPARISONS (same file vs same file):')
print('-'*70)
print(f'Count: {len(self_comp)}')
print(f'These are speakers with only one distance (0m):')
print(f'  - F3: {len(self_comp[self_comp["speaker_id"] == "F3"])} self-comparisons')
print(f'  - M2: {len(self_comp[self_comp["speaker_id"] == "M2"])} self-comparisons')
print(f'  - Others at 0.5m/1m: {len(self_comp) - len(self_comp[self_comp["speaker_id"].isin(["F3", "M2"])])}')

# Save cleaned results
real_comp.to_csv('results_bonafide/real_comparisons.csv', index=False)
print(f'\nSaved real comparisons to: results_bonafide/real_comparisons.csv')
