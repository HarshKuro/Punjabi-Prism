"""
Reporting and Visualization Module
==================================

Publication-quality visualizations and comprehensive reports.

Features:
- PESQ heatmaps by distance/device/speaker
- SNR distribution analysis
- Quality degradation curves
- Interactive HTML dashboard
- PDF report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3
})


class QualityReporter:
    """
    Generate comprehensive quality reports and visualizations.
    """
    
    def __init__(self, results_df: pd.DataFrame, output_dir: str = "reports"):
        """
        Initialize reporter.
        
        Args:
            results_df: DataFrame with pipeline results
            output_dir: Output directory for reports
        """
        self.df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = sns.color_palette("husl", 8)
        sns.set_style("whitegrid")
    
    # =========================================================================
    # HEATMAPS
    # =========================================================================
    
    def plot_pesq_heatmap_distance_speaker(self, save: bool = True) -> plt.Figure:
        """
        Create PESQ heatmap: Distance × Speaker.
        """
        if 'pesq' not in self.df.columns or 'distance_m' not in self.df.columns:
            logger.warning("Required columns not found for heatmap")
            return None
        
        # Pivot table
        pivot = self.df.pivot_table(
            values='pesq',
            index='speaker_id',
            columns='distance_m',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=2.5,
            vmin=1.0,
            vmax=4.5,
            ax=ax,
            cbar_kws={'label': 'PESQ Score'}
        )
        
        ax.set_title('PESQ Scores by Distance and Speaker', fontsize=16, pad=20)
        ax.set_xlabel('Distance (meters)', fontsize=14)
        ax.set_ylabel('Speaker ID', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "heatmap_pesq_distance_speaker.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_snr_heatmap_distance_speaker(self, save: bool = True) -> plt.Figure:
        """
        Create SNR heatmap: Distance × Speaker.
        """
        if 'snr_global_db' not in self.df.columns:
            logger.warning("SNR column not found for heatmap")
            return None
        
        # Pivot table
        pivot = self.df.pivot_table(
            values='snr_global_db',
            index='speaker_id',
            columns='distance_m',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            ax=ax,
            cbar_kws={'label': 'SNR (dB)'}
        )
        
        ax.set_title('SNR by Distance and Speaker', fontsize=16, pad=20)
        ax.set_xlabel('Distance (meters)', fontsize=14)
        ax.set_ylabel('Speaker ID', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "heatmap_snr_distance_speaker.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_metrics_heatmap_device_distance(self, metric: str = 'pesq',
                                              save: bool = True) -> plt.Figure:
        """
        Create metric heatmap: Device × Distance.
        """
        if metric not in self.df.columns or 'device' not in self.df.columns:
            logger.warning(f"Required columns not found for {metric} heatmap")
            return None
        
        # Pivot table
        pivot = self.df.pivot_table(
            values=metric,
            index='device',
            columns='distance_m',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select colormap based on metric
        if metric == 'pesq':
            cmap = 'RdYlGn'
            fmt = '.2f'
        else:
            cmap = 'viridis'
            fmt = '.1f'
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': metric.upper()}
        )
        
        ax.set_title(f'{metric.upper()} by Device and Distance', fontsize=16, pad=20)
        ax.set_xlabel('Distance (meters)', fontsize=14)
        ax.set_ylabel('Device', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"heatmap_{metric}_device_distance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # DISTRIBUTION PLOTS
    # =========================================================================
    
    def plot_quality_by_distance(self, save: bool = True) -> plt.Figure:
        """
        Plot quality metrics vs distance with error bars.
        """
        if 'distance_m' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        metrics = [
            ('pesq', 'PESQ Score', (1, 4.5)),
            ('snr_global_db', 'Global SNR (dB)', None),
            ('snr_segmental_db', 'Segmental SNR (dB)', None),
            ('stoi', 'STOI', (0, 1))
        ]
        
        for ax, (metric, label, ylim) in zip(axes.flat, metrics):
            if metric not in self.df.columns:
                ax.set_visible(False)
                continue
            
            # Group by distance
            grouped = self.df.groupby('distance_m')[metric].agg(['mean', 'std', 'count'])
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Plot with error bars
            ax.errorbar(
                grouped.index,
                grouped['mean'],
                yerr=1.96 * grouped['se'],  # 95% CI
                fmt='o-',
                capsize=5,
                capthick=2,
                markersize=8,
                linewidth=2,
                color=self.colors[0]
            )
            
            # Add individual points
            ax.scatter(
                self.df['distance_m'],
                self.df[metric],
                alpha=0.3,
                s=20,
                color=self.colors[1]
            )
            
            ax.set_xlabel('Distance (meters)', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label} vs Distance', fontsize=14)
            
            if ylim:
                ax.set_ylim(ylim)
            
            ax.grid(True, alpha=0.3)
            
            # Add sample sizes
            for idx, row in grouped.iterrows():
                ax.annotate(
                    f'n={int(row["count"])}',
                    (idx, row['mean']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9
                )
        
        plt.suptitle('Audio Quality Degradation with Distance', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "quality_by_distance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_metric_distributions(self, save: bool = True) -> plt.Figure:
        """
        Plot distributions of all quality metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('pesq', 'PESQ Score'),
            ('snr_global_db', 'Global SNR (dB)'),
            ('snr_segmental_db', 'Segmental SNR (dB)'),
            ('stoi', 'STOI')
        ]
        
        for ax, (metric, label) in zip(axes.flat, metrics):
            if metric not in self.df.columns:
                ax.set_visible(False)
                continue
            
            data = self.df[metric].dropna()
            
            # Histogram with KDE
            sns.histplot(data, kde=True, ax=ax, color=self.colors[0])
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--',
                      label=f'Median: {median_val:.2f}')
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Distribution of {label}', fontsize=14)
            ax.legend()
        
        plt.suptitle('Quality Metric Distributions', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "metric_distributions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_speaker_comparison(self, save: bool = True) -> plt.Figure:
        """
        Compare quality metrics across speakers.
        """
        if 'speaker_id' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # PESQ by speaker
        if 'pesq' in self.df.columns:
            ax = axes[0]
            sns.boxplot(data=self.df, x='speaker_id', y='pesq', ax=ax)
            sns.stripplot(data=self.df, x='speaker_id', y='pesq', 
                         ax=ax, alpha=0.5, size=4)
            ax.set_xlabel('Speaker ID', fontsize=12)
            ax.set_ylabel('PESQ Score', fontsize=12)
            ax.set_title('PESQ by Speaker', fontsize=14)
        
        # SNR by speaker
        if 'snr_global_db' in self.df.columns:
            ax = axes[1]
            sns.boxplot(data=self.df, x='speaker_id', y='snr_global_db', ax=ax)
            sns.stripplot(data=self.df, x='speaker_id', y='snr_global_db',
                         ax=ax, alpha=0.5, size=4)
            ax.set_xlabel('Speaker ID', fontsize=12)
            ax.set_ylabel('SNR (dB)', fontsize=12)
            ax.set_title('SNR by Speaker', fontsize=14)
        
        plt.suptitle('Quality Comparison Across Speakers', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "speaker_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, save: bool = True) -> plt.Figure:
        """
        Plot correlation matrix of quality metrics.
        """
        # Select numeric columns related to quality
        metric_cols = [col for col in self.df.columns if any(
            m in col for m in ['pesq', 'snr', 'stoi', 'correlation']
        )]
        
        if len(metric_cols) < 2:
            return None
        
        # Compute correlation
        corr = self.df[metric_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        
        ax.set_title('Correlation Matrix of Quality Metrics', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "correlation_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def generate_all_plots(self):
        """
        Generate all standard plots.
        """
        logger.info("Generating all plots...")
        
        self.plot_pesq_heatmap_distance_speaker()
        self.plot_snr_heatmap_distance_speaker()
        self.plot_metrics_heatmap_device_distance('pesq')
        self.plot_quality_by_distance()
        self.plot_metric_distributions()
        self.plot_speaker_comparison()
        self.plot_correlation_matrix()
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary report.
        """
        report = []
        report.append("=" * 70)
        report.append("AUDIO QUALITY ASSESSMENT REPORT")
        report.append("Samsung Prism - Punjabi Speech Dataset")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total files processed: {len(self.df)}")
        report.append(f"Successful: {self.df['success'].sum()}")
        report.append(f"Failed: {(~self.df['success']).sum()}")
        
        if 'speaker_id' in self.df.columns:
            report.append(f"Unique speakers: {self.df['speaker_id'].nunique()}")
        if 'distance_m' in self.df.columns:
            report.append(f"Distance range: {self.df['distance_m'].min()}m - {self.df['distance_m'].max()}m")
        
        report.append("")
        
        # PESQ summary
        if 'pesq' in self.df.columns:
            pesq_data = self.df['pesq'].dropna()
            report.append("PESQ SCORES (ITU-T P.862)")
            report.append("-" * 40)
            report.append(f"Mean:   {pesq_data.mean():.3f}")
            report.append(f"Std:    {pesq_data.std():.3f}")
            report.append(f"Min:    {pesq_data.min():.3f}")
            report.append(f"Max:    {pesq_data.max():.3f}")
            report.append(f"Median: {pesq_data.median():.3f}")
            report.append("")
            
            # By distance
            if 'distance_m' in self.df.columns:
                report.append("PESQ by Distance:")
                for dist in sorted(self.df['distance_m'].unique()):
                    subset = self.df[self.df['distance_m'] == dist]['pesq'].dropna()
                    if len(subset) > 0:
                        report.append(f"  {dist}m: {subset.mean():.3f} ± {subset.std():.3f} (n={len(subset)})")
            report.append("")
        
        # SNR summary
        if 'snr_global_db' in self.df.columns:
            snr_data = self.df['snr_global_db'].dropna()
            report.append("SNR (Global)")
            report.append("-" * 40)
            report.append(f"Mean:   {snr_data.mean():.1f} dB")
            report.append(f"Std:    {snr_data.std():.1f} dB")
            report.append(f"Min:    {snr_data.min():.1f} dB")
            report.append(f"Max:    {snr_data.max():.1f} dB")
            report.append("")
        
        # Quality assessment
        report.append("QUALITY ASSESSMENT")
        report.append("-" * 40)
        if 'pesq' in self.df.columns:
            excellent = (self.df['pesq'] >= 4.0).sum()
            good = ((self.df['pesq'] >= 3.0) & (self.df['pesq'] < 4.0)).sum()
            fair = ((self.df['pesq'] >= 2.0) & (self.df['pesq'] < 3.0)).sum()
            poor = (self.df['pesq'] < 2.0).sum()
            
            total = excellent + good + fair + poor
            if total > 0:
                report.append(f"Excellent (≥4.0): {excellent} ({100*excellent/total:.1f}%)")
                report.append(f"Good (3.0-4.0):   {good} ({100*good/total:.1f}%)")
                report.append(f"Fair (2.0-3.0):   {fair} ({100*fair/total:.1f}%)")
                report.append(f"Poor (<2.0):      {poor} ({100*poor/total:.1f}%)")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved to {report_path}")
        
        return report_text
    
    def generate_html_dashboard(self) -> str:
        """
        Generate interactive HTML dashboard.
        """
        # Get summary stats
        stats = {}
        
        if 'pesq' in self.df.columns:
            stats['pesq_mean'] = f"{self.df['pesq'].mean():.3f}"
            stats['pesq_std'] = f"{self.df['pesq'].std():.3f}"
        
        if 'snr_global_db' in self.df.columns:
            stats['snr_mean'] = f"{self.df['snr_global_db'].mean():.1f}"
        
        stats['total_files'] = len(self.df)
        stats['success_rate'] = f"{100 * self.df['success'].mean():.1f}%"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Quality Dashboard - Samsung Prism</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #1a73e8; color: white; padding: 20px; border-radius: 8px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #1a73e8; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .plot-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-card img {{ max-width: 100%; height: auto; }}
        h2 {{ color: #333; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Audio Quality Dashboard</h1>
        <p>Samsung Prism - Punjabi Speech Quality Assessment</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['total_files']}</div>
            <div class="stat-label">Total Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get('pesq_mean', 'N/A')}</div>
            <div class="stat-label">Mean PESQ</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats.get('snr_mean', 'N/A')} dB</div>
            <div class="stat-label">Mean SNR</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['success_rate']}</div>
            <div class="stat-label">Success Rate</div>
        </div>
    </div>
    
    <h2>Quality Analysis</h2>
    <div class="plot-grid">
        <div class="plot-card">
            <h3>PESQ by Distance & Speaker</h3>
            <img src="heatmap_pesq_distance_speaker.png" alt="PESQ Heatmap">
        </div>
        <div class="plot-card">
            <h3>Quality Degradation with Distance</h3>
            <img src="quality_by_distance.png" alt="Quality by Distance">
        </div>
        <div class="plot-card">
            <h3>Metric Distributions</h3>
            <img src="metric_distributions.png" alt="Distributions">
        </div>
        <div class="plot-card">
            <h3>Correlation Matrix</h3>
            <img src="correlation_matrix.png" alt="Correlations">
        </div>
    </div>
    
    <h2>Speaker Analysis</h2>
    <div class="plot-card">
        <img src="speaker_comparison.png" alt="Speaker Comparison">
    </div>
    
    <footer style="margin-top: 40px; padding: 20px; background: #333; color: white; border-radius: 8px;">
        <p>Pipeline Configuration Hash: [config_hash]</p>
        <p>ITU-T P.862 Compliant PESQ | Validated SNR Computation</p>
    </footer>
</body>
</html>
"""
        
        # Save HTML
        html_path = self.output_dir / "dashboard.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML dashboard saved to {html_path}")
        
        return str(html_path)
    
    def generate_full_report(self):
        """
        Generate complete report package.
        """
        logger.info("Generating full report package...")
        
        # Generate all plots
        self.generate_all_plots()
        
        # Generate text summary
        self.generate_summary_report()
        
        # Generate HTML dashboard
        self.generate_html_dashboard()
        
        # Export filtered data
        success_df = self.df[self.df['success'] == True]
        success_df.to_csv(self.output_dir / "successful_results.csv", index=False)
        
        logger.info(f"Full report package saved to {self.output_dir}")


def generate_report(results_csv: str, output_dir: str = "reports"):
    """
    Convenience function to generate report from CSV.
    
    Args:
        results_csv: Path to results CSV file
        output_dir: Output directory for reports
    """
    df = pd.read_csv(results_csv)
    reporter = QualityReporter(df, output_dir)
    reporter.generate_full_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reporting.py <results.csv> [output_dir]")
        sys.exit(1)
    
    results_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    
    generate_report(results_csv, output_dir)
