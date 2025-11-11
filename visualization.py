"""
Visualization Module
===================

Publication-quality visualizations for audio quality research
with proper error bars, confidence intervals, and statistical annotations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings

from perfect_audio_quality import logger

# Configure matplotlib for publication quality
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class ResearchVisualizer:
    """Publication-quality visualizations for audio quality research"""
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'husl'):
        """
        Initialize visualizer
        
        Args:
            style: Seaborn style
            palette: Color palette
        """
        sns.set_style(style)
        sns.set_palette(palette)
        self.colors = sns.color_palette(palette, 10)
        
    def plot_distance_quality_relationship(self, df: pd.DataFrame, 
                                         quality_metric: str = 'pesq_score',
                                         distance_col: str = 'distance_meters',
                                         uncertainty_col: str = None,
                                         save_path: str = None) -> plt.Figure:
        """
        Plot quality vs distance with confidence intervals
        
        Args:
            df: DataFrame with results
            quality_metric: Quality metric column name
            distance_col: Distance column name
            uncertainty_col: Uncertainty column name
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by distance and calculate statistics
        grouped = df.groupby(distance_col)[quality_metric].agg([
            'mean', 'std', 'count', 'median',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75)
        ]).reset_index()
        
        grouped.columns = [distance_col, 'mean', 'std', 'count', 'median', 'q1', 'q3']
        
        # Calculate confidence intervals
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['se']
        grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['se']
        
        # Plot individual points with transparency
        ax.scatter(df[distance_col], df[quality_metric], 
                  alpha=0.3, s=30, color=self.colors[0], label='Individual measurements')
        
        # Plot means with error bars
        ax.errorbar(grouped[distance_col], grouped['mean'],
                   yerr=[grouped['mean'] - grouped['ci_lower'], 
                         grouped['ci_upper'] - grouped['mean']],
                   fmt='o', capsize=5, capthick=2, markersize=8,
                   color=self.colors[1], label='Mean ± 95% CI')
        
        # Add uncertainty bars if available
        if uncertainty_col and uncertainty_col in df.columns:
            for _, row in grouped.iterrows():
                subset = df[df[distance_col] == row[distance_col]]
                if len(subset) > 0:
                    mean_uncertainty = subset[uncertainty_col].mean()
                    ax.errorbar(row[distance_col], row['mean'],
                              yerr=mean_uncertainty,
                              fmt='s', capsize=3, capthick=1, markersize=6,
                              color=self.colors[2], alpha=0.7)
        
        # Fit and plot trend line
        if len(grouped) > 2:
            # Log transformation for distance (distance + 1 to handle 0)
            x_trend = np.log(grouped[distance_col] + 1)
            y_trend = grouped['mean']
            
            # Fit polynomial
            coeffs = np.polyfit(x_trend, y_trend, 1)
            x_smooth = np.linspace(x_trend.min(), x_trend.max(), 100)
            y_smooth = np.poly1d(coeffs)(x_smooth)
            
            # Convert back to original scale
            x_smooth_orig = np.exp(x_smooth) - 1
            
            ax.plot(x_smooth_orig, y_smooth, '--', 
                   color=self.colors[3], linewidth=2, 
                   label=f'Trend: y = {coeffs[0]:.3f}×ln(x+1) + {coeffs[1]:.3f}')
            
            # Calculate R²
            y_pred = np.poly1d(coeffs)(x_trend)
            r_squared = 1 - np.sum((y_trend - y_pred)**2) / np.sum((y_trend - y_trend.mean())**2)
            
            # Add R² to plot
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Distance from Source (meters)', fontsize=14)
        ax.set_ylabel(self._format_metric_name(quality_metric), fontsize=14)
        ax.set_title(f'{self._format_metric_name(quality_metric)} vs Recording Distance', 
                    fontsize=16, pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add sample sizes as annotations
        for _, row in grouped.iterrows():
            ax.annotate(f'n={int(row["count"])}', 
                       (row[distance_col], row['mean']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distance-quality plot saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                               metrics: List[str] = None,
                               method: str = 'pearson',
                               save_path: str = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            df: DataFrame with results
            metrics: List of metrics to include
            method: Correlation method
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            # Auto-detect quality metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if any(
                metric in col.lower() for metric in ['pesq', 'snr', 'stoi']
            )]
        
        # Calculate correlation matrix
        corr_data = df[metrics].corr(method=method)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f',
                   center=0, cmap='RdBu_r', vmin=-1, vmax=1,
                   square=True, ax=ax, cbar_kws={"shrink": .8})
        
        # Customize
        ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=16, pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Format tick labels
        ax.set_xticklabels([self._format_metric_name(col) for col in corr_data.columns],
                          rotation=45, ha='right')
        ax.set_yticklabels([self._format_metric_name(col) for col in corr_data.columns],
                          rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        return fig
    
    def plot_quality_distributions(self, df: pd.DataFrame,
                                  metrics: List[str] = None,
                                  group_by: str = None,
                                  save_path: str = None) -> plt.Figure:
        """
        Plot quality metric distributions
        
        Args:
            df: DataFrame with results
            metrics: List of metrics to plot
            group_by: Column to group by
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if any(
                metric in col.lower() for metric in ['pesq', 'snr', 'stoi']
            )][:3]  # Limit to 3 metrics
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if group_by and group_by in df.columns:
                # Box plot with groups
                sns.boxplot(data=df, x=group_by, y=metric, ax=ax)
                
                # Add individual points
                sns.stripplot(data=df, x=group_by, y=metric, 
                             size=4, alpha=0.6, ax=ax)
                
                ax.set_xlabel(self._format_metric_name(group_by))
            else:
                # Histogram with KDE
                sns.histplot(data=df, x=metric, kde=True, ax=ax)
                
                # Add mean and median lines
                mean_val = df[metric].mean()
                median_val = df[metric].median()
                
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle='--', 
                          label=f'Median: {median_val:.3f}')
                ax.legend()
            
            ax.set_ylabel(self._format_metric_name(metric))
            ax.set_title(f'{self._format_metric_name(metric)} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {save_path}")
        
        return fig
    
    def plot_regression_diagnostics(self, df: pd.DataFrame,
                                   dependent_var: str,
                                   independent_var: str,
                                   save_path: str = None) -> plt.Figure:
        """
        Plot regression diagnostics
        
        Args:
            df: DataFrame with data
            dependent_var: Dependent variable
            independent_var: Independent variable
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # Prepare data
        clean_df = df[[dependent_var, independent_var]].dropna()
        X = clean_df[[independent_var]]
        y = clean_df[dependent_var]
        
        # Fit regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        residuals = y - y_pred
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Regression plot with confidence interval
        ax1 = axes[0, 0]
        sns.regplot(data=clean_df, x=independent_var, y=dependent_var, 
                   ax=ax1, scatter_kws={'alpha': 0.6})
        ax1.set_title('Regression Fit with Confidence Interval')
        ax1.grid(True, alpha=0.3)
        
        # Add R² to plot
        r_squared = lr.score(X, y)
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals vs Fitted
        ax2 = axes[0, 1]
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Fitted Values')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot for residual normality
        ax3 = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Residual Normality)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scale-Location plot
        ax4 = axes[1, 1]
        standardized_residuals = residuals / np.std(residuals)
        ax4.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
        ax4.set_xlabel('Fitted Values')
        ax4.set_ylabel('√|Standardized Residuals|')
        ax4.set_title('Scale-Location Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regression diagnostics saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame,
                                   save_path: str = None) -> go.Figure:
        """
        Create interactive Plotly dashboard
        
        Args:
            df: DataFrame with results
            save_path: Path to save HTML file
            
        Returns:
            Plotly Figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality vs Distance', 'Metric Correlations',
                          'Quality Distributions', 'Speaker Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Quality vs Distance scatter plot
        if 'pesq_score' in df.columns and 'distance_meters' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['distance_meters'],
                    y=df['pesq_score'],
                    mode='markers',
                    name='PESQ',
                    text=df['speaker_id'] if 'speaker_id' in df.columns else None,
                    hovertemplate='Distance: %{x}m<br>PESQ: %{y:.3f}<br>Speaker: %{text}',
                ),
                row=1, col=1
            )
        
        # 2. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quality_metrics = [col for col in numeric_cols if any(
            metric in col.lower() for metric in ['pesq', 'snr', 'stoi']
        )][:5]  # Limit to 5 metrics
        
        if len(quality_metrics) > 1:
            corr_matrix = df[quality_metrics].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=[self._format_metric_name(col) for col in corr_matrix.columns],
                    y=[self._format_metric_name(col) for col in corr_matrix.columns],
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation'
                ),
                row=1, col=2
            )
        
        # 3. Quality distributions
        if quality_metrics:
            for i, metric in enumerate(quality_metrics[:3]):  # Show top 3
                fig.add_trace(
                    go.Histogram(
                        x=df[metric],
                        name=self._format_metric_name(metric),
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # 4. Speaker analysis
        if 'speaker_id' in df.columns and 'pesq_score' in df.columns:
            speaker_stats = df.groupby('speaker_id')['pesq_score'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=speaker_stats['speaker_id'],
                    y=speaker_stats['mean'],
                    error_y=dict(type='data', array=speaker_stats['std']),
                    name='Mean PESQ by Speaker'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Audio Quality Analysis Dashboard',
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Distance (m)", row=1, col=1)
        fig.update_yaxes(title_text="PESQ Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Quality Metric", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Speaker ID", row=2, col=2)
        fig.update_yaxes(title_text="Mean PESQ Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric names for display"""
        name_mapping = {
            'pesq_score': 'PESQ Score',
            'global_snr_score': 'Global SNR (dB)',
            'segmental_snr_score': 'Segmental SNR (dB)',
            'perceptual_snr_score': 'Perceptual SNR (dB)',
            'stoi_score': 'STOI Score',
            'distance_meters': 'Distance (m)',
            'speaker_id': 'Speaker ID',
            'gender': 'Gender'
        }
        
        return name_mapping.get(metric_name, metric_name.replace('_', ' ').title())

# Example usage
if __name__ == "__main__":
    visualizer = ResearchVisualizer()
    logger.info("Research Visualizer initialized")
    logger.info("Ready to create publication-quality plots")
