#!/usr/bin/env python3
"""
Script to regenerate sector charts using existing data without API calls.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_existing_results():
    """Load existing beta results from CSV."""
    try:
        df = pd.read_csv('sp500_optimized_results.csv', index_col=0)
        # Convert to dictionary format
        beta_results = {}
        for symbol, row in df.iterrows():
            # Skip ticker K
            if symbol == 'K':
                continue
            beta_results[symbol] = {
                'traditional_beta': row['traditional_beta'],
                'positive_beta': row['positive_beta'],
                'negative_beta': row['negative_beta'],
                'beta_ratio': row['beta_ratio']
            }
        print(f"Loaded {len(beta_results)} companies from existing results (excluding K)")
        return beta_results
    except FileNotFoundError:
        print("No existing results found.")
        return {}

def load_sector_map():
    """Load sector mapping from Wikipedia data."""
    try:
        df = pd.read_csv('sp500_wikipedia_data.csv')
        sector_map = dict(zip(df['symbol'], df['sector']))
        return sector_map
    except FileNotFoundError:
        print("Wikipedia data not found.")
        return {}

def create_sector_charts(beta_results, sector_map):
    """Create sector-by-sector bar charts split into two files for better readability."""
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    
    # Add sector
    df['sector'] = df.index.map(sector_map)
    
    # Filter for valid beta ratios
    df = df.dropna(subset=['beta_ratio'])
    
    # Get unique sectors
    sectors = df['sector'].unique()
    
    # Split sectors into two groups
    first_6_sectors = sectors[:6]
    remaining_5_sectors = sectors[6:11]
    
    # Create first chart with 6 sectors (3x2 layout) - much taller for better readability
    fig1, axes1 = plt.subplots(3, 2, figsize=(20, 48))  # Increased height from 36 to 48
    axes1 = axes1.flatten()
    
    for i, sector in enumerate(first_6_sectors):
        ax = axes1[i]
        sector_data = df[df['sector'] == sector].sort_values('beta_ratio', ascending=True)
        
        if len(sector_data) == 0:
            continue
            
        # Create horizontal bar chart with gradient color scheme (red at top, blue at bottom)
        colors = []
        for ratio in sector_data['beta_ratio']:
            if ratio < 0.8:
                colors.append('blue')  # Highest negative bias at bottom
            elif ratio < 0.9:
                colors.append('lightblue')
            elif ratio < 1.0:
                colors.append('yellow')
            elif ratio < 1.1:
                colors.append('orange')
            else:
                colors.append('red')  # Highest positive bias at top
        
        bars = ax.barh(range(len(sector_data)), sector_data['beta_ratio'], color=colors)
        
        # Add company labels with better spacing for readability
        ax.set_yticks(range(len(sector_data)))
        ax.set_yticklabels(sector_data.index, fontsize=8)  # Slightly larger font
        
        # Add reference line at 1.0
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector title
        ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Beta Ratio (Positive/Negative)')
        
        # Add value labels on bars
        for j, (bar, ratio) in enumerate(zip(bars, sector_data['beta_ratio'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.3f}', va='center', fontsize=7)  # Slightly larger font for value labels
    
    # Hide unused subplot
    axes1[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/sp500_sector_charts_part1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("First 6 sectors chart saved to docs/sp500_sector_charts_part1.png")
    
    # Create second chart with remaining 5 sectors (3x2 layout) - much taller for better readability
    fig2, axes2 = plt.subplots(3, 2, figsize=(20, 48))  # Increased height from 36 to 48
    axes2 = axes2.flatten()
    
    for i, sector in enumerate(remaining_5_sectors):
        ax = axes2[i]
        sector_data = df[df['sector'] == sector].sort_values('beta_ratio', ascending=True)
        
        if len(sector_data) == 0:
            continue
            
        # Create horizontal bar chart with gradient color scheme (red at top, blue at bottom)
        colors = []
        for ratio in sector_data['beta_ratio']:
            if ratio < 0.8:
                colors.append('blue')  # Highest negative bias at bottom
            elif ratio < 0.9:
                colors.append('lightblue')
            elif ratio < 1.0:
                colors.append('yellow')
            elif ratio < 1.1:
                colors.append('orange')
            else:
                colors.append('red')  # Highest positive bias at top
        
        bars = ax.barh(range(len(sector_data)), sector_data['beta_ratio'], color=colors)
        
        # Add company labels with better spacing for readability
        ax.set_yticks(range(len(sector_data)))
        ax.set_yticklabels(sector_data.index, fontsize=8)  # Slightly larger font
        
        # Add reference line at 1.0
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector title
        ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Beta Ratio (Positive/Negative)')
        
        # Add value labels on bars
        for j, (bar, ratio) in enumerate(zip(bars, sector_data['beta_ratio'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.3f}', va='center', fontsize=7)  # Slightly larger font for value labels
    
    # Hide unused subplot
    axes2[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/sp500_sector_charts_part2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Remaining 5 sectors chart saved to docs/sp500_sector_charts_part2.png")

def create_sector_beta_charts(beta_results, sector_map):
    """Create bar charts showing positive vs negative betas by sector with 90% confidence intervals."""
    if not beta_results:
        print("No beta results available.")
        return
    
    # Group by sector
    sector_data = {}
    for symbol, result in beta_results.items():
        if symbol in sector_map and result['positive_beta'] is not None and result['negative_beta'] is not None:
            sector = sector_map[symbol]
            if sector not in sector_data:
                sector_data[sector] = {'positive': [], 'negative': []}
            sector_data[sector]['positive'].append(result['positive_beta'])
            sector_data[sector]['negative'].append(result['negative_beta'])
    
    # Calculate sector averages and confidence intervals
    sector_averages = {}
    for sector, data in sector_data.items():
        if len(data['positive']) > 0 and len(data['negative']) > 0:
            # Calculate means
            pos_mean = np.mean(data['positive'])
            neg_mean = np.mean(data['negative'])
            
            # Calculate 90% confidence intervals
            pos_ci = 1.645 * np.std(data['positive']) / np.sqrt(len(data['positive']))  # 90% CI
            neg_ci = 1.645 * np.std(data['negative']) / np.sqrt(len(data['negative']))  # 90% CI
            
            sector_averages[sector] = {
                'positive_avg': pos_mean,
                'negative_avg': neg_mean,
                'positive_ci': pos_ci,
                'negative_ci': neg_ci,
                'ratio': pos_mean / neg_mean
            }
    
    # Sort sectors by ratio
    sorted_sectors = sorted(sector_averages.items(), key=lambda x: x[1]['ratio'])
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    sectors = [s[0] for s in sorted_sectors]
    positive_avgs = [s[1]['positive_avg'] for s in sorted_sectors]
    negative_avgs = [s[1]['negative_avg'] for s in sorted_sectors]
    positive_cis = [s[1]['positive_ci'] for s in sorted_sectors]
    negative_cis = [s[1]['negative_ci'] for s in sorted_sectors]
    ratios = [s[1]['ratio'] for s in sorted_sectors]
    
    # Bar chart with confidence intervals
    x = np.arange(len(sectors))
    width = 0.35
    
    # Use original nice colors: green for positive, red for negative
    bars1 = ax1.bar(x - width/2, positive_avgs, width, label='Positive Beta', color='green', alpha=0.7, yerr=positive_cis, capsize=5)
    bars2 = ax1.bar(x + width/2, negative_avgs, width, label='Negative Beta', color='red', alpha=0.7, yerr=negative_cis, capsize=5)
    
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Average Beta')
    ax1.set_title('Average Positive vs Negative Betas by Sector (with 90% Confidence Intervals)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sectors, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Ratio chart with original nice color scheme
    colors = ['green' if r >= 1.0 else 'red' for r in ratios]
    bars3 = ax2.bar(sectors, ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Symmetric (1.0)')
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Beta Ratio (Positive/Negative)')
    ax2.set_title('Beta Asymmetry by Sector')
    ax2.set_xticklabels(sectors, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on ratio bars
    for bar, ratio in zip(bars3, ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/sector_beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sector beta comparison chart saved to docs/sector_beta_comparison.png")

def create_beta_comparison_chart(beta_results, sector_map):
    """Create beta comparison chart showing positive vs negative betas for individual stocks."""
    if not beta_results:
        print("No beta results available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    df = df.dropna(subset=['positive_beta', 'negative_beta'])
    
    # Sort by positive beta for better visualization
    df_sorted = df.sort_values('positive_beta', ascending=False)
    
    # Take top 20 stocks for readability
    top_stocks = df_sorted.head(20)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(top_stocks))
    width = 0.35
    
    # Create bars for positive and negative betas
    bars1 = ax.bar(x - width/2, top_stocks['positive_beta'], width, label='Positive Beta', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, top_stocks['negative_beta'], width, label='Negative Beta', color='red', alpha=0.7)
    
    ax.set_xlabel('Stock')
    ax.set_ylabel('Beta Value')
    ax.set_title('Positive vs Negative Betas for Top 20 Stocks')
    ax.set_xticks(x)
    ax.set_xticklabels(top_stocks.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Beta comparison chart saved to docs/beta_comparison.png")

def create_comprehensive_sector_chart(beta_results, sector_map):
    """Create comprehensive sector chart with 4-4-3 layout and much taller height."""
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    
    # Add sector
    df['sector'] = df.index.map(sector_map)
    
    # Filter for valid beta ratios
    df = df.dropna(subset=['beta_ratio'])
    
    # Get unique sectors
    sectors = df['sector'].unique()
    
    # Create comprehensive chart with 4-4-3 layout (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(24, 60))  # Much taller for better readability
    axes = axes.flatten()
    
    for i, sector in enumerate(sectors):
        if i >= len(axes):
            break
            
        ax = axes[i]
        sector_data = df[df['sector'] == sector].sort_values('beta_ratio', ascending=True)
        
        if len(sector_data) == 0:
            continue
            
        # Create horizontal bar chart with gradient color scheme (red at top, blue at bottom)
        colors = []
        for ratio in sector_data['beta_ratio']:
            if ratio < 0.8:
                colors.append('blue')  # Highest negative bias at bottom
            elif ratio < 0.9:
                colors.append('lightblue')
            elif ratio < 1.0:
                colors.append('yellow')
            elif ratio < 1.1:
                colors.append('orange')
            else:
                colors.append('red')  # Highest positive bias at top
        
        bars = ax.barh(range(len(sector_data)), sector_data['beta_ratio'], color=colors)
        
        # Add company labels with better spacing for readability
        ax.set_yticks(range(len(sector_data)))
        ax.set_yticklabels(sector_data.index, fontsize=9)  # Larger font for better readability
        
        # Add reference line at 1.0
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector title
        ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Beta Ratio (Positive/Negative)')
        
        # Add value labels on bars
        for j, (bar, ratio) in enumerate(zip(bars, sector_data['beta_ratio'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.3f}', va='center', fontsize=8)  # Larger font for value labels
    
    # Hide unused subplots
    for i in range(len(sectors), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/sp500_optimized_sector_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comprehensive sector chart saved to docs/sp500_optimized_sector_charts.png")

def create_beta_comparison_charts(beta_results, sector_map):
    """Create beta comparison charts for highest and lowest beta stocks."""
    if not beta_results:
        print("No beta results available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    df = df.dropna(subset=['positive_beta', 'negative_beta'])
    
    # Calculate traditional beta for sorting
    df['traditional_beta'] = (df['positive_beta'] + df['negative_beta']) / 2
    
    # Sort by traditional beta
    df_sorted_high = df.sort_values('traditional_beta', ascending=False)
    df_sorted_low = df.sort_values('traditional_beta', ascending=True)  # This is already lowest to highest
    
    # Take top and bottom 20 stocks
    top_20_stocks = df_sorted_high.head(20)
    bottom_20_stocks = df_sorted_low.head(20)  # These are already sorted lowest to highest
    
    # Create chart for highest beta stocks
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(top_20_stocks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, top_20_stocks['positive_beta'], width, label='Positive Beta', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, top_20_stocks['negative_beta'], width, label='Negative Beta', color='red', alpha=0.7)
    
    ax1.set_xlabel('Stock')
    ax1.set_ylabel('Beta Value')
    ax1.set_title('Positive vs Negative Betas for 20 Highest Beta Stocks')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_20_stocks.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/beta_comparison_highest.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Highest beta comparison chart saved to docs/beta_comparison_highest.png")
    
    # Create chart for lowest beta stocks
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(bottom_20_stocks))
    width = 0.35
    
    bars3 = ax2.bar(x - width/2, bottom_20_stocks['positive_beta'], width, label='Positive Beta', color='green', alpha=0.7)
    bars4 = ax2.bar(x + width/2, bottom_20_stocks['negative_beta'], width, label='Negative Beta', color='red', alpha=0.7)
    
    ax2.set_xlabel('Stock')
    ax2.set_ylabel('Beta Value')
    ax2.set_title('Positive vs Negative Betas for 20 Lowest Beta Stocks')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bottom_20_stocks.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/beta_comparison_lowest.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Lowest beta comparison chart saved to docs/beta_comparison_lowest.png")

def create_scatter_plot(beta_results, sector_map):
    """Create scatter plot with positive over negative betas."""
    if not beta_results:
        print("No beta results available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    df = df.dropna(subset=['positive_beta', 'negative_beta'])
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all points
    ax.scatter(df['positive_beta'], df['negative_beta'], alpha=0.6, s=30)
    
    # Add diagonal line for equal betas
    max_val = max(df['positive_beta'].max(), df['negative_beta'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal betas')
    
    ax.set_xlabel('Positive Beta')
    ax.set_ylabel('Negative Beta')
    ax.set_title('Positive vs Negative Market Beta')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('docs/beta_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Beta scatter plot saved to docs/beta_scatter_plot.png")

def create_comprehensive_statistical_analysis(beta_results, sector_map):
    """Create comprehensive 2x2 statistical analysis chart."""
    if not beta_results:
        print("No beta results available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(beta_results, orient='index')
    df = df.dropna(subset=['positive_beta', 'negative_beta'])
    
    # Calculate beta differences
    df['beta_difference'] = df['positive_beta'] - df['negative_beta']
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Scatter plot of positive vs negative betas
    ax1.scatter(df['positive_beta'], df['negative_beta'], alpha=0.6, s=30)
    max_val = max(df['positive_beta'].max(), df['negative_beta'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal betas')
    ax1.set_xlabel('Positive Market Beta')
    ax1.set_ylabel('Negative Market Beta')
    ax1.set_title('Positive vs Negative Market Betas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Histogram of beta differences
    ax2.hist(df['beta_difference'], bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No difference')
    ax2.axvline(x=df['beta_difference'].mean(), color='green', linestyle='-', alpha=0.7, 
                label=f'Mean diff: {df["beta_difference"].mean():.4f}')
    ax2.set_xlabel('Beta Difference (Positive - Negative)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Beta Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Box plot of positive vs negative betas
    box_data = [df['positive_beta'], df['negative_beta']]
    ax3.boxplot(box_data, labels=['Positive Beta', 'Negative Beta'])
    ax3.set_ylabel('Beta Value')
    ax3.set_title('Box Plot: Positive vs Negative Betas')
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Top 10 stocks by absolute beta difference
    top_10_by_diff = df.nlargest(10, 'beta_difference')
    bottom_10_by_diff = df.nsmallest(10, 'beta_difference')
    
    # Combine and sort by absolute difference
    combined_diff = pd.concat([top_10_by_diff, bottom_10_by_diff])
    combined_diff['abs_diff'] = abs(combined_diff['beta_difference'])
    top_10_abs = combined_diff.nlargest(10, 'abs_diff')
    
    colors = ['blue' if x >= 0 else 'red' for x in top_10_abs['beta_difference']]
    bars = ax4.barh(range(len(top_10_abs)), top_10_abs['beta_difference'], color=colors, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_yticks(range(len(top_10_abs)))
    ax4.set_yticklabels(top_10_abs.index)
    ax4.set_xlabel('Beta Difference (Positive - Negative)')
    ax4.set_title('Top 10 Stocks by Beta Difference')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, diff in zip(bars, top_10_abs['beta_difference']):
        ax4.text(bar.get_width() + 0.01 if bar.get_width() >= 0 else bar.get_width() - 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{diff:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/comprehensive_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comprehensive statistical analysis saved to docs/comprehensive_statistical_analysis.png")

def main():
    """Main function to regenerate charts."""
    print("Regenerating charts using existing data...")
    
    # Load existing data
    beta_results = load_existing_results()
    sector_map = load_sector_map()
    
    if not beta_results:
        print("No existing results found. Please run the main analysis first.")
        return
    
    # Create output directory
    os.makedirs('docs', exist_ok=True)
    
    # Generate charts
    create_sector_charts(beta_results, sector_map)
    create_sector_beta_charts(beta_results, sector_map)
    create_beta_comparison_chart(beta_results, sector_map)
    create_comprehensive_sector_chart(beta_results, sector_map)
    create_beta_comparison_charts(beta_results, sector_map)
    create_scatter_plot(beta_results, sector_map)
    create_comprehensive_statistical_analysis(beta_results, sector_map)
    
    print("Chart regeneration complete!")

if __name__ == "__main__":
    main() 