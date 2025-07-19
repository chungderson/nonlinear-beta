#!/usr/bin/env python3
"""
Econometric analysis of asymmetric beta results.
Performs statistical tests to determine significance of findings.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the beta results data."""
    try:
        df = pd.read_csv('sp500_optimized_results.csv', index_col=0)
        # Exclude K
        df = df[df.index != 'K']
        print(f"Loaded {len(df)} companies for econometric analysis")
        return df
    except FileNotFoundError:
        print("No results data found. Please run the main analysis first.")
        return None

def perform_paired_ttest(df):
    """Perform paired t-test on positive vs negative betas."""
    print("\n" + "="*60)
    print("PAIRED T-TEST: POSITIVE VS NEGATIVE BETAS")
    print("="*60)
    
    # Remove any rows with NaN values
    clean_df = df.dropna(subset=['positive_beta', 'negative_beta'])
    
    positive_betas = clean_df['positive_beta']
    negative_betas = clean_df['negative_beta']
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(positive_betas, negative_betas)
    
    print(f"Sample size: {len(clean_df)}")
    print(f"Mean positive beta: {positive_betas.mean():.4f}")
    print(f"Mean negative beta: {negative_betas.mean():.4f}")
    print(f"Mean difference (positive - negative): {positive_betas.mean() - negative_betas.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    # Determine significance
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "Not significant (p >= 0.05)"
    
    print(f"Statistical significance: {significance}")
    
    # Calculate effect size (Cohen's d)
    differences = positive_betas - negative_betas
    effect_size = differences.mean() / differences.std()
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    
    return t_stat, p_value, effect_size

def test_sector_asymmetry(df):
    """Test for significant asymmetry within each sector."""
    print("\n" + "="*60)
    print("SECTOR-SPECIFIC ASYMMETRY TESTS")
    print("="*60)
    
    # Load sector mapping
    try:
        sector_df = pd.read_csv('sp500_wikipedia_data.csv')
        sector_map = dict(zip(sector_df['symbol'], sector_df['sector']))
        df['sector'] = df.index.map(sector_map)
    except FileNotFoundError:
        print("Sector data not found. Skipping sector analysis.")
        return
    
    # Group by sector and test each
    sector_results = []
    
    for sector in df['sector'].unique():
        if pd.isna(sector):
            continue
            
        sector_data = df[df['sector'] == sector]
        sector_data = sector_data.dropna(subset=['positive_beta', 'negative_beta'])
        
        if len(sector_data) < 5:  # Need minimum sample size
            continue
            
        positive_betas = sector_data['positive_beta']
        negative_betas = sector_data['negative_beta']
        
        # Paired t-test for this sector
        t_stat, p_value = stats.ttest_rel(positive_betas, negative_betas)
        
        mean_diff = positive_betas.mean() - negative_betas.mean()
        effect_size = mean_diff / (positive_betas - negative_betas).std()
        
        # Determine significance
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        sector_results.append({
            'Sector': sector,
            'N': len(sector_data),
            'Mean_Positive': positive_betas.mean(),
            'Mean_Negative': negative_betas.mean(),
            'Mean_Difference': mean_diff,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Significance': significance,
            'Effect_Size': effect_size
        })
    
    # Create results table
    results_df = pd.DataFrame(sector_results)
    results_df = results_df.sort_values('P_Value')
    
    print("\nSector Asymmetry Test Results:")
    print("-" * 100)
    print(f"{'Sector':<20} {'N':<5} {'Pos_Mean':<10} {'Neg_Mean':<10} {'Diff':<10} {'T-Stat':<10} {'P-Value':<10} {'Sig':<5} {'Effect':<10}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['Sector']:<20} {row['N']:<5} {row['Mean_Positive']:<10.3f} {row['Mean_Negative']:<10.3f} "
              f"{row['Mean_Difference']:<10.3f} {row['T_Statistic']:<10.3f} {row['P_Value']:<10.4f} "
              f"{row['Significance']:<5} {row['Effect_Size']:<10.3f}")
    
    return results_df

def test_extreme_cases(df):
    """Test statistical significance of extreme asymmetric cases."""
    print("\n" + "="*60)
    print("EXTREME ASYMMETRY CASE ANALYSIS")
    print("="*60)
    
    # Calculate beta differences
    df['beta_difference'] = df['positive_beta'] - df['negative_beta']
    df['abs_beta_difference'] = abs(df['beta_difference'])
    
    # Find top 10 most asymmetric stocks
    top_asymmetric = df.nlargest(10, 'abs_beta_difference')
    
    print("\nTop 10 Most Asymmetric Stocks:")
    print("-" * 80)
    print(f"{'Stock':<8} {'Pos_Beta':<10} {'Neg_Beta':<10} {'Difference':<12} {'Ratio':<10}")
    print("-" * 80)
    
    for stock, row in top_asymmetric.iterrows():
        ratio = row['positive_beta'] / row['negative_beta']
        print(f"{stock:<8} {row['positive_beta']:<10.3f} {row['negative_beta']:<10.3f} "
              f"{row['beta_difference']:<12.3f} {ratio:<10.3f}")
    
    # Test if these extreme cases are significantly different from the population
    population_mean_diff = df['beta_difference'].mean()
    population_std_diff = df['beta_difference'].std()
    
    print(f"\nPopulation mean difference: {population_mean_diff:.4f}")
    print(f"Population std difference: {population_std_diff:.4f}")
    
    # Z-test for each extreme case
    print("\nStatistical Significance of Extreme Cases (Z-test):")
    print("-" * 80)
    print(f"{'Stock':<8} {'Difference':<12} {'Z-Score':<10} {'P-Value':<10} {'Significance':<15}")
    print("-" * 80)
    
    for stock, row in top_asymmetric.iterrows():
        z_score = (row['beta_difference'] - population_mean_diff) / population_std_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "ns (p >= 0.05)"
        
        print(f"{stock:<8} {row['beta_difference']:<12.3f} {z_score:<10.3f} {p_value:<10.4f} {significance:<15}")
    
    return top_asymmetric

def perform_regression_analysis(df):
    """Perform regression analysis to test for systematic patterns."""
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    # Clean data
    clean_df = df.dropna(subset=['positive_beta', 'negative_beta', 'traditional_beta'])
    
    # Test if traditional beta is a good predictor of asymmetry
    clean_df['asymmetry_ratio'] = clean_df['positive_beta'] / clean_df['negative_beta']
    clean_df['beta_difference'] = clean_df['positive_beta'] - clean_df['negative_beta']
    
    # Regression 1: Traditional beta vs asymmetry ratio
    print("\nRegression 1: Traditional Beta vs Asymmetry Ratio")
    print("-" * 50)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_df['traditional_beta'], clean_df['asymmetry_ratio']
    )
    
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R-squared: {r_value**2:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Standard error: {std_err:.6f}")
    
    # Regression 2: Traditional beta vs beta difference
    print("\nRegression 2: Traditional Beta vs Beta Difference")
    print("-" * 50)
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        clean_df['traditional_beta'], clean_df['beta_difference']
    )
    
    print(f"Slope: {slope2:.6f}")
    print(f"Intercept: {intercept2:.6f}")
    print(f"R-squared: {r_value2**2:.6f}")
    print(f"P-value: {p_value2:.6f}")
    print(f"Standard error: {std_err2:.6f}")

def main():
    """Main econometric analysis function."""
    print("ECONOMETRIC ANALYSIS OF ASYMMETRIC BETA RESULTS")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Perform analyses
    t_stat, p_value, effect_size = perform_paired_ttest(df)
    sector_results = test_sector_asymmetry(df)
    extreme_cases = test_extreme_cases(df)
    perform_regression_analysis(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ECONOMETRIC FINDINGS")
    print("="*60)
    
    if p_value < 0.05:
        print("✓ Overall asymmetric beta effect is statistically significant")
        print(f"  - T-statistic: {t_stat:.4f}")
        print(f"  - P-value: {p_value:.6f}")
        print(f"  - Effect size: {effect_size:.4f}")
    else:
        print("✗ Overall asymmetric beta effect is not statistically significant")
        print(f"  - P-value: {p_value:.6f}")
    
    # Count significant sectors
    significant_sectors = sector_results[sector_results['P_Value'] < 0.05]
    print(f"\n✓ {len(significant_sectors)} sectors show significant asymmetry")
    
    # Count extreme cases
    extreme_significant = 0
    for _, case in extreme_cases.iterrows():
        z_score = (case['beta_difference'] - df['beta_difference'].mean()) / df['beta_difference'].std()
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        if p_val < 0.05:
            extreme_significant += 1
    
    print(f"✓ {extreme_significant} out of 10 extreme cases are statistically significant")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 