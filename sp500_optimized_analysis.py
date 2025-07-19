#!/usr/bin/env python3
"""
Optimized S&P 500 Beta Asymmetry Analysis
Better rate limiting and progress tracking.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
from scipy import stats
import warnings
import concurrent.futures
import threading
from queue import Queue
import pickle
import os
warnings.filterwarnings('ignore')

from getBars import getBars
from helperMethods import getTradingDays, calculateDrift

def get_sp500_from_wikipedia():
    """
    Get all S&P 500 companies from Wikipedia table with GICS sectors.
    """
    try:
        # Try to load from the scraped CSV file
        df = pd.read_csv('sp500_wikipedia_data.csv')
        
        # Create symbol to sector mapping
        sector_map = {}
        symbols = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            sector = row['sector']
            
            if symbol and sector:
                sector_map[symbol] = sector
                symbols.append(symbol)
        
        print(f"Loaded {len(symbols)} S&P 500 companies from Wikipedia data")
        return symbols, sector_map
        
    except FileNotFoundError:
        print("Wikipedia data file not found. Please run sp500_wikipedia_scraper.py first.")
        return [], {}

def calculate_beta_clean(stock_data, market_data):
    """
    Calculate betas cleanly: regression of stock returns vs market returns.
    """
    if stock_data is None or market_data is None:
        return None
    
    # Calculate returns
    stock_returns = stock_data['close'].pct_change().dropna()
    market_returns = market_data['close'].pct_change().dropna()
    
    # Align data
    aligned_data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned_data) < 50:
        return None
    
    # Calculate traditional beta (regression coefficient)
    # β = Cov(Ri, Rm) / Var(Rm)
    covariance = np.cov(aligned_data['stock'], aligned_data['market'])[0,1]
    market_variance = np.var(aligned_data['market'])
    
    if market_variance == 0:
        return None
    
    traditional_beta = covariance / market_variance
    
    # Split by positive/negative market days
    positive_days = aligned_data[aligned_data['market'] > 0]
    negative_days = aligned_data[aligned_data['market'] < 0]
    
    # Calculate positive beta (regression on positive market days)
    if len(positive_days) > 20:
        pos_covariance = np.cov(positive_days['stock'], positive_days['market'])[0,1]
        pos_market_variance = np.var(positive_days['market'])
        if pos_market_variance != 0:
            positive_beta = pos_covariance / pos_market_variance
        else:
            positive_beta = None
    else:
        positive_beta = None
    
    # Calculate negative beta (regression on negative market days)
    if len(negative_days) > 20:
        neg_covariance = np.cov(negative_days['stock'], negative_days['market'])[0,1]
        neg_market_variance = np.var(negative_days['market'])
        if neg_market_variance != 0:
            negative_beta = neg_covariance / neg_market_variance
        else:
            negative_beta = None
    else:
        negative_beta = None
    
    # Calculate beta ratio
    if positive_beta is not None and negative_beta is not None and negative_beta != 0:
        beta_ratio = positive_beta / negative_beta
    else:
        beta_ratio = None
    
    return {
        'traditional_beta': traditional_beta,
        'positive_beta': positive_beta,
        'negative_beta': negative_beta,
        'beta_ratio': beta_ratio,
        'data_points': len(aligned_data),
        'positive_days': len(positive_days),
        'negative_days': len(negative_days)
    }

class OptimizedRateLimiter:
    """Optimized rate limiter with better batching."""
    def __init__(self, max_calls=200, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit with better timing."""
        with self.lock:
            now = time.time()
            
            # Remove calls older than time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.wait_if_needed()
            
            # Add current call
            self.calls.append(now)
            self.last_call_time = now

def fetch_single_stock(symbol, start_date, end_date, rate_limiter):
    """Fetch data for a single stock with rate limiting."""
    rate_limiter.wait_if_needed()
    
    try:
        data = getBars(symbol, start_date, end_date)
        if data is not None and len(data) > 0:
            return symbol, data
        else:
            return symbol, None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return symbol, None

class SP500OptimizedAnalyzer:
    def __init__(self):
        self.results = {}
        self.market_data = None
        self.rate_limiter = OptimizedRateLimiter(max_calls=200, time_window=60)
        self.progress_file = 'sp500_progress.pkl'
        
    def load_progress(self):
        """Load progress from file if it exists."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f:
                    progress = pickle.load(f)
                    self.results = progress.get('results', {})
                    self.market_data = progress.get('market_data', None)
                    print(f"Loaded progress: {len(self.results)} stocks already fetched")
                    return True
            except Exception as e:
                print(f"Error loading progress: {e}")
        return False
    
    def save_progress(self):
        """Save progress to file."""
        try:
            progress = {
                'results': self.results,
                'market_data': self.market_data
            }
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress, f)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def fetch_data_optimized(self, symbols, start_date='2021-01-01', end_date=None, max_workers=3):
        """Fetch data for all symbols with optimized rate limiting."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Load existing progress
        self.load_progress()
        
        # Filter out already fetched symbols
        remaining_symbols = [s for s in symbols if s not in self.results]
        
        if not remaining_symbols:
            print("All symbols already fetched!")
            return True
            
        print(f"Fetching data for {len(remaining_symbols)} remaining S&P 500 companies...")
        
        # Fetch market data (SPY) if not already loaded
        if self.market_data is None:
            print("Fetching market data (SPY)...")
            self.rate_limiter.wait_if_needed()
            self.market_data = getBars('SPY', start_date, end_date)
            if self.market_data is None:
                print("Failed to fetch market data")
                return False
            print(f"✓ Market data: {len(self.market_data)} days")
        
        # Process in smaller batches with 5-second delays
        batch_size = 20  # Smaller batches
        total_batches = (len(remaining_symbols) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_symbols))
            batch_symbols = remaining_symbols[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)")
            
            # Fetch data for batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(fetch_single_stock, symbol, start_date, end_date, self.rate_limiter): symbol
                    for symbol in batch_symbols
                }
                
                # Collect results as they complete
                batch_completed = 0
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol, data = future.result()
                    if data is not None:
                        self.results[symbol] = data
                        batch_completed += 1
                    
                    # Save progress every 5 symbols
                    if batch_completed % 5 == 0:
                        self.save_progress()
                        print(f"  ✓ Batch progress: {batch_completed}/{len(batch_symbols)}")
            
            print(f"  ✓ Batch {batch_num + 1} complete: {batch_completed}/{len(batch_symbols)} symbols fetched")
            
            # 5-second delay between batches
            if batch_num < total_batches - 1:  # Don't delay after the last batch
                print("  Waiting 5 seconds before next batch...")
                time.sleep(5)
        
        print(f"\nData collection complete. Market: {len(self.market_data)} days, Stocks: {len(self.results)}")
        
        # Clean up progress file
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        
        return True
    
    def calculate_betas(self):
        """Calculate traditional and nonlinear betas for all stocks."""
        if not self.results or self.market_data is None:
            print("No data available. Run fetch_data_optimized() first.")
            return
            
        print("\nCalculating beta relationships...")
        
        beta_results = {}
        
        for symbol, data in self.results.items():
            if len(data) < 100:
                continue
                
            # Calculate betas using clean approach
            results = calculate_beta_clean(data, self.market_data)
            
            if results is not None:
                beta_results[symbol] = results
                print(f"{symbol}: Trad β={results['traditional_beta']:.3f}, Pos β={results['positive_beta']:.3f}, Neg β={results['negative_beta']:.3f}, Ratio={results['beta_ratio']:.3f}")
        
        return beta_results
    
    def create_sector_beta_charts(self, beta_results, sector_map):
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
                    'positive_std': np.std(data['positive']),
                    'negative_std': np.std(data['negative']),
                    'count': len(data['positive'])
                }
        
        if not sector_averages:
            print("No valid sector data found.")
            return
        
        # Sort sectors by positive beta average
        sorted_sectors = sorted(sector_averages.items(), key=lambda x: x[1]['positive_avg'], reverse=True)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Extract data for plotting
        sectors = [s[0] for s in sorted_sectors]
        positive_avgs = [s[1]['positive_avg'] for s in sorted_sectors]
        negative_avgs = [s[1]['negative_avg'] for s in sorted_sectors]
        positive_stds = [s[1]['positive_std'] for s in sorted_sectors]
        negative_stds = [s[1]['negative_std'] for s in sorted_sectors]
        counts = [s[1]['count'] for s in sorted_sectors]
        
        x = np.arange(len(sectors))
        width = 0.35
        
        # Plot 1: Positive vs Negative Beta Averages
        bars1 = ax1.bar(x - width/2, positive_avgs, width, label='Positive Beta', 
                        color='green', alpha=0.7, yerr=positive_stds, capsize=5)
        bars2 = ax1.bar(x + width/2, negative_avgs, width, label='Negative Beta', 
                        color='red', alpha=0.7, yerr=negative_stds, capsize=5)
        
        ax1.set_xlabel('Sector')
        ax1.set_ylabel('Beta Value')
        ax1.set_title('Average Positive vs Negative Betas by Sector')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sectors, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (pos, neg) in enumerate(zip(positive_avgs, negative_avgs)):
            ax1.text(i - width/2, pos + 0.02, f'{pos:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width/2, neg + 0.02, f'{neg:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Beta Ratio by Sector
        ratios = [pos/neg if neg != 0 else 0 for pos, neg in zip(positive_avgs, negative_avgs)]
        colors = ['green' if r > 1 else 'red' if r < 1 else 'gray' for r in ratios]
        
        bars3 = ax2.bar(x, ratios, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Symmetric (Ratio = 1)')
        
        ax2.set_xlabel('Sector')
        ax2.set_ylabel('Beta Ratio (Positive/Negative)')
        ax2.set_title('Beta Ratio by Sector (Positive Beta / Negative Beta)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sectors, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, ratio in enumerate(ratios):
            ax2.text(i, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax2.text(i, 0.1, f'n={count}', ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('docs', exist_ok=True)
        plt.savefig('docs/sector_beta_comparison.png', dpi=300, bbox_inches='tight')
        print("Sector beta comparison charts saved to docs/sector_beta_comparison.png")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SECTOR BETA SUMMARY")
        print("="*80)
        print(f"{'Sector':<25} {'Pos Avg':<10} {'Neg Avg':<10} {'Ratio':<10} {'Count':<8}")
        print("-"*80)
        for sector, data in sorted_sectors:
            ratio = data['positive_avg'] / data['negative_avg'] if data['negative_avg'] != 0 else 0
            print(f"{sector:<25} {data['positive_avg']:<10.3f} {data['negative_avg']:<10.3f} {ratio:<10.3f} {data['count']:<8}")
        
        plt.show()

    def create_sector_charts(self, beta_results, sector_map):
        """Create sector-by-sector bar charts with all companies properly plotted."""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(beta_results, orient='index')
        
        # Add sector
        df['sector'] = df.index.map(sector_map)
        
        # Filter for valid beta ratios
        df = df.dropna(subset=['beta_ratio'])
        
        # Get unique sectors
        sectors = df['sector'].unique()
        
        # Create subplots for each sector
        n_sectors = len(sectors)
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        for i, sector in enumerate(sectors):
            if i >= len(axes):
                break
                
            sector_data = df[df['sector'] == sector].sort_values('beta_ratio', ascending=True)
            
            if len(sector_data) == 0:
                continue
                
            ax = axes[i]
            
            # Create bar chart with all companies in the sector
            y_positions = range(len(sector_data))
            bars = ax.barh(y_positions, sector_data['beta_ratio'], 
                          color=['red' if x > 1.1 else 'orange' if x > 1.0 else 'yellow' if x > 0.9 else 'lightblue' if x > 0.8 else 'blue' for x in sector_data['beta_ratio']])
            
            # Add reference line at 1.0
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Set labels
            ax.set_yticks(y_positions)
            ax.set_yticklabels(sector_data.index, fontsize=8)
            ax.set_xlabel('Positive/Negative Beta Ratio')
            ax.set_title(f'{sector} ({len(sector_data)} companies)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add ratio values on bars
            for j, (idx, row) in enumerate(sector_data.iterrows()):
                ax.text(row['beta_ratio'], j, f'{row["beta_ratio"]:.3f}', 
                       ha='left', va='center', fontsize=6)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_sectors, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('docs/sp500_optimized_sector_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\nOptimized sector charts saved to docs/sp500_optimized_sector_charts.png")
        
        return df
    
    def create_ascending_list(self, beta_results, sector_map):
        """Create a comprehensive list of all companies in ascending order."""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(beta_results, orient='index')
        
        # Add sector
        df['sector'] = df.index.map(sector_map)
        
        # Filter for valid beta ratios
        df = df.dropna(subset=['beta_ratio'])
        
        # Sort by beta ratio (ascending)
        df_sorted = df.sort_values('beta_ratio', ascending=True)
        
        # Create the comprehensive list
        print("\n" + "="*100)
        print("OPTIMIZED S&P 500 COMPANIES BY BETA RATIO (ASCENDING ORDER)")
        print("="*100)
        
        print(f"{'Rank':<4} {'Symbol':<8} {'Ratio':<8} {'Sector':<25} {'Trad Beta':<10} {'Pos Beta':<10} {'Neg Beta':<10}")
        print("-"*100)
        
        for i, (symbol, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i:<4} {symbol:<8} {row['beta_ratio']:<8.3f} {row['sector']:<25} {row['traditional_beta']:<10.3f} {row['positive_beta']:<10.3f} {row['negative_beta']:<10.3f}")
        
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total Companies Analyzed: {len(df_sorted)}")
        print(f"Mean Beta Ratio: {df_sorted['beta_ratio'].mean():.3f}")
        print(f"Median Beta Ratio: {df_sorted['beta_ratio'].median():.3f}")
        print(f"Standard Deviation: {df_sorted['beta_ratio'].std():.3f}")
        print(f"Companies with Ratio > 1.0: {(df_sorted['beta_ratio'] > 1.0).sum()}")
        print(f"Companies with Ratio < 1.0: {(df_sorted['beta_ratio'] < 1.0).sum()}")
        
        # Sector breakdown
        print("\nSECTOR BREAKDOWN:")
        sector_stats = df_sorted.groupby('sector')['beta_ratio'].agg(['count', 'mean', 'median'])
        for sector, stats in sector_stats.iterrows():
            print(f"{sector:<25}: {stats['count']:>3} companies, Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
        
        return df_sorted

def main():
    """Main function to run the optimized S&P 500 analysis."""
    analyzer = SP500OptimizedAnalyzer()
    
    # Get S&P 500 symbols with proper sector classification
    symbols, sector_map = get_sp500_from_wikipedia()
    
    if not symbols:
        print("No S&P 500 symbols found. Please run sp500_wikipedia_scraper.py first.")
        return
    
    # Fetch data with optimized approach
    if not analyzer.fetch_data_optimized(symbols, max_workers=3):
        print("Failed to fetch data")
        return
    
    # Calculate betas
    beta_results = analyzer.calculate_betas()
    
    if not beta_results:
        print("No valid beta results")
        return
    
    # Create sector beta comparison charts
    analyzer.create_sector_beta_charts(beta_results, sector_map)
    
    # Create sector charts
    df = analyzer.create_sector_charts(beta_results, sector_map)
    
    # Create ascending list
    df_sorted = analyzer.create_ascending_list(beta_results, sector_map)
    
    # Save results
    df_sorted.to_csv('sp500_optimized_results.csv')
    print(f"\nOptimized results saved to 'sp500_optimized_results.csv'")
    
    return analyzer

if __name__ == "__main__":
    main() 