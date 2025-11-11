"""
Statistical Analysis Module
==========================

Comprehensive statistical analysis for audio quality research
with proper significance testing, effect sizes, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

from perfect_audio_quality import logger

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for audio quality research"""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical analyzer
        
        Args:
            alpha: Significance level for hypothesis tests
            power: Desired statistical power
        """
        self.alpha = alpha
        self.power = power
        
        # Configure plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
    def descriptive_statistics(self, df: pd.DataFrame, group_by: str = None) -> Dict:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            df: DataFrame with quality measurements
            group_by: Column to group by (e.g., 'distance_meters')
            
        Returns:
            Dictionary with descriptive statistics
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        quality_metrics = [col for col in numeric_columns if any(
            metric in col.lower() for metric in ['pesq', 'snr', 'stoi']
        )]
        
        results = {}
        
        if group_by and group_by in df.columns:
            # Grouped descriptive statistics
            for metric in quality_metrics:
                if metric in df.columns:
                    grouped_stats = df.groupby(group_by)[metric].agg([
                        'count', 'mean', 'std', 'min', 'max', 'median',
                        lambda x: np.percentile(x, 25),  # Q1
                        lambda x: np.percentile(x, 75),  # Q3
                        'skew', 'kurtosis'
                    ]).round(4)
                    
                    grouped_stats.columns = [
                        'count', 'mean', 'std', 'min', 'max', 'median', 
                        'q1', 'q3', 'skewness', 'kurtosis'
                    ]
                    
                    results[f'{metric}_by_{group_by}'] = grouped_stats.to_dict('index')
        else:
            # Overall descriptive statistics
            for metric in quality_metrics:
                if metric in df.columns:
                    data = df[metric].dropna()
                    
                    results[metric] = {
                        'count': len(data),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'median': float(data.median()),
                        'q1': float(data.quantile(0.25)),
                        'q3': float(data.quantile(0.75)),
                        'skewness': float(data.skew()),
                        'kurtosis': float(data.kurtosis()),
                        'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else np.nan
                    }
        
        return results
    
    def test_normality(self, df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Test normality of data distributions
        
        Args:
            df: DataFrame with quality measurements
            metrics: List of metric columns to test
            
        Returns:
            Dictionary with normality test results
        """
        results = {}
        
        for metric in metrics:
            if metric in df.columns:
                data = df[metric].dropna()
                
                if len(data) < 8:
                    results[metric] = {'error': 'Insufficient data for normality tests'}
                    continue
                
                # Shapiro-Wilk test (best for small samples)
                if len(data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    shapiro_result = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > self.alpha
                    }
                else:
                    shapiro_result = {'note': 'Skipped - sample too large for Shapiro-Wilk'}
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                ks_result = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > self.alpha
                }
                
                # Anderson-Darling test
                ad_result = stats.anderson(data, dist='norm')
                ad_critical_value = ad_result.critical_values[2]  # 5% significance level
                ad_is_normal = ad_result.statistic < ad_critical_value
                
                results[metric] = {
                    'shapiro_wilk': shapiro_result,
                    'kolmogorov_smirnov': ks_result,
                    'anderson_darling': {
                        'statistic': float(ad_result.statistic),
                        'critical_value_5pct': float(ad_critical_value),
                        'is_normal': ad_is_normal
                    },
                    'recommendation': 'normal' if all([
                        shapiro_result.get('is_normal', True),
                        ks_result['is_normal'],
                        ad_is_normal
                    ]) else 'non_normal'
                }
        
        return results
    
    def anova_analysis(self, df: pd.DataFrame, dependent_var: str, 
                      independent_var: str) -> Dict:
        """
        Perform one-way ANOVA with post-hoc tests
        
        Args:
            df: DataFrame with data
            dependent_var: Dependent variable (e.g., 'pesq_score')
            independent_var: Independent variable (e.g., 'distance_meters')
            
        Returns:
            Dictionary with ANOVA results
        """
        # Prepare data
        clean_df = df[[dependent_var, independent_var]].dropna()
        groups = [group[dependent_var].values for name, group in clean_df.groupby(independent_var)]
        group_names = [name for name, group in clean_df.groupby(independent_var)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for ANOVA'}
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        total_sum_squares = np.sum([(x - clean_df[dependent_var].mean())**2 
                                   for group in groups for x in group])
        between_sum_squares = np.sum([len(group) * (np.mean(group) - clean_df[dependent_var].mean())**2 
                                     for group in groups])
        eta_squared = between_sum_squares / total_sum_squares
        
        # Post-hoc tests (Tukey HSD)
        posthoc_results = {}
        if p_value < self.alpha:
            from scipy.stats import tukey_hsd
            
            # Pairwise comparisons
            for i, group1_name in enumerate(group_names):
                for j, group2_name in enumerate(group_names):
                    if i < j:  # Avoid duplicate comparisons
                        group1_data = groups[i]
                        group2_data = groups[j]
                        
                        # Tukey HSD
                        tukey_result = tukey_hsd(group1_data, group2_data)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                            (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                                           (len(group1_data) + len(group2_data) - 2))
                        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
                        
                        posthoc_results[f'{group1_name}_vs_{group2_name}'] = {
                            'tukey_hsd_pvalue': float(tukey_result.pvalue),
                            'mean_difference': float(np.mean(group1_data) - np.mean(group2_data)),
                            'cohens_d': float(cohens_d),
                            'significant': tukey_result.pvalue < self.alpha,
                            'effect_size_interpretation': self._interpret_cohens_d(abs(cohens_d))
                        }
        
        return {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'eta_squared': float(eta_squared),
                'effect_size_interpretation': self._interpret_eta_squared(eta_squared)
            },
            'group_statistics': {
                str(name): {
                    'n': len(group),
                    'mean': float(np.mean(group)),
                    'std': float(np.std(group, ddof=1)),
                    'median': float(np.median(group))
                } for name, group in zip(group_names, groups)
            },
            'posthoc_tests': posthoc_results
        }
    
    def correlation_analysis(self, df: pd.DataFrame, 
                           variables: List[str]) -> Dict:
        """
        Comprehensive correlation analysis
        
        Args:
            df: DataFrame with data
            variables: List of variables to correlate
            
        Returns:
            Dictionary with correlation results
        """
        # Clean data
        clean_df = df[variables].dropna()
        
        if len(clean_df) < 3:
            return {'error': 'Insufficient data for correlation analysis'}
        
        results = {}
        
        # Correlation matrix
        correlation_methods = {
            'pearson': clean_df.corr(method='pearson'),
            'spearman': clean_df.corr(method='spearman'),
            'kendall': clean_df.corr(method='kendall')
        }
        
        # Detailed pairwise correlations
        pairwise_results = {}
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Avoid duplicate pairs
                    data1 = clean_df[var1]
                    data2 = clean_df[var2]
                    
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(data1, data2)
                    
                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(data1, data2)
                    
                    # Kendall correlation
                    kendall_r, kendall_p = kendalltau(data1, data2)
                    
                    # Bootstrap confidence intervals for Pearson
                    ci_lower, ci_upper = self._bootstrap_correlation_ci(data1, data2)
                    
                    pairwise_results[f'{var1}_vs_{var2}'] = {
                        'pearson': {
                            'correlation': float(pearson_r),
                            'p_value': float(pearson_p),
                            'significant': pearson_p < self.alpha,
                            'confidence_interval': (float(ci_lower), float(ci_upper)),
                            'r_squared': float(pearson_r**2)
                        },
                        'spearman': {
                            'correlation': float(spearman_r),
                            'p_value': float(spearman_p),
                            'significant': spearman_p < self.alpha
                        },
                        'kendall': {
                            'correlation': float(kendall_r),
                            'p_value': float(kendall_p),
                            'significant': kendall_p < self.alpha
                        }
                    }
        
        return {
            'correlation_matrices': {method: matrix.to_dict() 
                                   for method, matrix in correlation_methods.items()},
            'pairwise_correlations': pairwise_results,
            'sample_size': len(clean_df)
        }
    
    def regression_analysis(self, df: pd.DataFrame, dependent_var: str, 
                          independent_vars: List[str], robust: bool = True) -> Dict:
        """
        Comprehensive regression analysis with validation
        
        Args:
            df: DataFrame with data
            dependent_var: Dependent variable
            independent_vars: List of independent variables
            robust: Whether to use robust regression
            
        Returns:
            Dictionary with regression results
        """
        # Prepare data
        variables = [dependent_var] + independent_vars
        clean_df = df[variables].dropna()
        
        if len(clean_df) < len(independent_vars) + 5:  # Need sufficient observations
            return {'error': 'Insufficient data for regression analysis'}
        
        X = clean_df[independent_vars]
        y = clean_df[dependent_var]
        
        results = {}
        
        # Standard linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        
        # Robust regression
        if robust:
            robust_lr = HuberRegressor()
            robust_lr.fit(X, y)
            y_pred_robust = robust_lr.predict(X)
        
        # Model performance metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
        
        # Residual analysis
        residuals = y - y_pred
        
        results['linear_regression'] = {
            'coefficients': {var: float(coef) for var, coef in zip(independent_vars, lr.coef_)},
            'intercept': float(lr.intercept_),
            'r_squared': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'cross_validation': {
                'mean_r2': float(cv_scores.mean()),
                'std_r2': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
        }
        
        if robust:
            r2_robust = r2_score(y, y_pred_robust)
            results['robust_regression'] = {
                'coefficients': {var: float(coef) for var, coef in zip(independent_vars, robust_lr.coef_)},
                'intercept': float(robust_lr.intercept_),
                'r_squared': float(r2_robust),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred_robust))),
                'mae': float(mean_absolute_error(y, y_pred_robust))
            }
        
        # Residual analysis
        results['residual_analysis'] = {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'residual_normality': self._test_residual_normality(residuals),
            'durbin_watson': self._durbin_watson_test(residuals)
        }
        
        return results
    
    def power_analysis(self, effect_size: float, alpha: float = None, 
                      power: float = None, n: int = None) -> Dict:
        """
        Statistical power analysis
        
        Args:
            effect_size: Expected effect size (Cohen's d or eta-squared)
            alpha: Significance level
            power: Desired power
            n: Sample size
            
        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        if power is None:
            power = self.power
        
        # This is a simplified power analysis
        # For more complex analyses, consider using statsmodels or specialized libraries
        
        results = {}
        
        if n is None:
            # Calculate required sample size
            # Using simplified formula for t-test
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n_required = 2 * ((z_alpha + z_beta) / effect_size)**2
            results['required_sample_size'] = int(np.ceil(n_required))
        
        if n is not None:
            # Calculate achieved power
            delta = effect_size * np.sqrt(n/2)
            achieved_power = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - delta)
            results['achieved_power'] = float(achieved_power)
        
        results['inputs'] = {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'sample_size': n
        }
        
        return results
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_func: callable,
                                    n_bootstrap: int = 10000,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for any statistic
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _bootstrap_correlation_ci(self, x: np.ndarray, y: np.ndarray, 
                                 n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence interval for correlation"""
        bootstrap_correlations = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(x), size=len(x), replace=True)
            x_boot = x.iloc[indices] if hasattr(x, 'iloc') else x[indices]
            y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            corr, _ = pearsonr(x_boot, y_boot)
            bootstrap_correlations.append(corr)
        
        ci_lower = np.percentile(bootstrap_correlations, 2.5)
        ci_upper = np.percentile(bootstrap_correlations, 97.5)
        
        return ci_lower, ci_upper
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def _test_residual_normality(self, residuals: np.ndarray) -> Dict:
        """Test normality of residuals"""
        if len(residuals) < 8:
            return {'error': 'Insufficient data'}
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(residuals)
        
        return {
            'shapiro_wilk_statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value > self.alpha
        }
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        n = len(residuals)
        if n < 2:
            return np.nan
        
        diff_residuals = np.diff(residuals)
        dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        return float(dw)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer()
    logger.info("Statistical Analyzer initialized")
    logger.info(f"Significance level: {analyzer.alpha}")
    logger.info(f"Desired power: {analyzer.power}")
