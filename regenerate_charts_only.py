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
    
    # Create first chart with 6 sectors (3x2 layout) - taller for better readability
    fig1, axes1 = plt.subplots(3, 2, figsize=(20, 30))  # Increased height from 24 to 30
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
        ax.set_yticklabels(sector_data.index, fontsize=7)  # Slightly smaller font
        
        # Add reference line at 1.0
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector title
        ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Beta Ratio (Positive/Negative)')
        
        # Add value labels on bars
        for j, (bar, ratio) in enumerate(zip(bars, sector_data['beta_ratio'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.3f}', va='center', fontsize=6)  # Smaller font for value labels
    
    # Hide unused subplot
    axes1[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/sp500_sector_charts_part1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("First 6 sectors chart saved to docs/sp500_sector_charts_part1.png")
    
    # Create second chart with remaining 5 sectors (3x2 layout) - taller for better readability
    fig2, axes2 = plt.subplots(3, 2, figsize=(20, 30))  # Increased height from 24 to 30
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
        ax.set_yticklabels(sector_data.index, fontsize=7)  # Slightly smaller font
        
        # Add reference line at 1.0
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector title
        ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Beta Ratio (Positive/Negative)')
        
        # Add value labels on bars
        for j, (bar, ratio) in enumerate(zip(bars, sector_data['beta_ratio'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.3f}', va='center', fontsize=6)  # Smaller font for value labels
    
    # Hide unused subplot
    axes2[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('docs/sp500_sector_charts_part2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Remaining 5 sectors chart saved to docs/sp500_sector_charts_part2.png")

def create_sector_beta_charts(beta_results, sector_map):
    """Create bar charts showing positive vs negative betas by sector."""
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
    
    # Calculate sector averages
    sector_averages = {}
    for sector, data in sector_data.items():
        if len(data['positive']) > 0 and len(data['negative']) > 0:
            sector_averages[sector] = {
                'positive_avg': np.mean(data['positive']),
                'negative_avg': np.mean(data['negative']),
                'ratio': np.mean(data['positive']) / np.mean(data['negative'])
            }
    
    # Sort sectors by ratio
    sorted_sectors = sorted(sector_averages.items(), key=lambda x: x[1]['ratio'])
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    sectors = [s[0] for s in sorted_sectors]
    positive_avgs = [s[1]['positive_avg'] for s in sorted_sectors]
    negative_avgs = [s[1]['negative_avg'] for s in sorted_sectors]
    ratios = [s[1]['ratio'] for s in sorted_sectors]
    
    # Bar chart with gradient colors
    x = np.arange(len(sectors))
    width = 0.35
    
    # Use gradient colors for positive/negative bars too (red at top, blue at bottom)
    pos_colors = []
    neg_colors = []
    for ratio in ratios:
        if ratio < 0.8:
            pos_colors.append('blue')  # Highest negative bias at bottom
            neg_colors.append('blue')
        elif ratio < 0.9:
            pos_colors.append('lightblue')
            neg_colors.append('lightblue')
        elif ratio < 1.0:
            pos_colors.append('yellow')
            neg_colors.append('yellow')
        elif ratio < 1.1:
            pos_colors.append('orange')
            neg_colors.append('orange')
        else:
            pos_colors.append('red')  # Highest positive bias at top
            neg_colors.append('red')
    
    bars1 = ax1.bar(x - width/2, positive_avgs, width, label='Positive Beta', color=pos_colors, alpha=0.7)
    bars2 = ax1.bar(x + width/2, negative_avgs, width, label='Negative Beta', color=neg_colors, alpha=0.7)
    
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Average Beta')
    ax1.set_title('Average Positive vs Negative Betas by Sector')
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
    
    # Ratio chart with gradient color scheme (red at top, blue at bottom)
    colors = []
    for ratio in ratios:
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
    
    print("Chart regeneration complete!")

if __name__ == "__main__":
    main() 