#!/usr/bin/env python3
"""
S&P 500 Beta Asymmetry Heatmap
Creates a heatmap showing positive-to-negative beta ratios for S&P 500 companies,
weighted by market capitalization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import your existing Alpaca functions
from getBars import getBars
from helperMethods import getTradingDays, calculateDrift

def load_config():
    """Load API configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config['ALPACA_KEY'], config['ALPACA_SECRET']
    except FileNotFoundError:
        print("config.json not found. Please create it with your Alpaca API credentials.")
        return None, None

def get_sp500_symbols():
    """
    Get S&P 500 symbols. For now, we'll use a subset of major companies.
    In a full implementation, you could fetch the complete S&P 500 list from an API.
    """
    # Major S&P 500 companies by market cap (top 50)
    sp500_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK.B', 'LLY', 'TSLA', 'UNH',
        'V', 'XOM', 'JNJ', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
        'PEP', 'KO', 'AVGO', 'COST', 'MRK', 'PFE', 'TMO', 'ACN', 'DHR', 'NEE',
        'ABT', 'TXN', 'QCOM', 'HON', 'NKE', 'PM', 'ORCL', 'IBM', 'AMD', 'INTC',
        'VZ', 'CMCSA', 'ADP', 'UNP', 'RTX', 'LOW', 'SPGI', 'ISRG', 'INTU', 'ADI'
    ]
    return sp500_symbols

def get_market_cap_data(symbols):
    """
    Get market cap data for symbols. For now, we'll use approximate market caps.
    In a full implementation, you could fetch real-time market cap data.
    """
    # Approximate market caps (in billions) - you could fetch this from an API
    market_caps = {
        'AAPL': 3000, 'MSFT': 2800, 'GOOGL': 1800, 'AMZN': 1800, 'NVDA': 1200,
        'META': 1000, 'BRK.B': 800, 'LLY': 700, 'TSLA': 600, 'UNH': 500,
        'V': 500, 'XOM': 450, 'JNJ': 400, 'WMT': 500, 'JPM': 400, 'PG': 350,
        'MA': 400, 'HD': 350, 'CVX': 300, 'ABBV': 250, 'PEP': 250, 'KO': 250,
        'AVGO': 200, 'COST': 300, 'MRK': 250, 'PFE': 150, 'TMO': 200, 'ACN': 200,
        'DHR': 200, 'NEE': 150, 'ABT': 200, 'TXN': 150, 'QCOM': 200, 'HON': 150,
        'NKE': 200, 'PM': 150, 'ORCL': 300, 'IBM': 150, 'AMD': 200, 'INTC': 200,
        'VZ': 150, 'CMCSA': 200, 'ADP': 100, 'UNP': 150, 'RTX': 100, 'LOW': 150,
        'SPGI': 100, 'ISRG': 100, 'INTU': 150, 'ADI': 100
    }
    
    return {symbol: market_caps.get(symbol, 50) for symbol in symbols}

class SP500BetaAnalyzer:
    def __init__(self):
        self.results = {}
        self.market_data = None
        
    def fetch_data(self, symbols, start_date='2021-01-01', end_date=None):
        """Fetch data for all symbols and market index."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {len(symbols)} S&P 500 companies...")
        
        # Fetch market data (SPY)
        print("Fetching market data (SPY)...")
        self.market_data = getBars('SPY', start_date, end_date)
        if self.market_data is None:
            print("Failed to fetch market data")
            return False
            
        print(f"✓ Market data: {len(self.market_data)} days")
        
        # Fetch data for each symbol
        for i, symbol in enumerate(symbols):
            print(f"Fetching data for {symbol}...")
            data = getBars(symbol, start_date, end_date)
            
            if data is not None and len(data) > 0:
                self.results[symbol] = data
                print(f"✓ {symbol}: {len(data)} days of data")
            else:
                print(f"✗ {symbol}: No data available")
                
            # Rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(1)
                
        print(f"Data collection complete. Market: {len(self.market_data)} days")
        return True
    
    def calculate_betas(self):
        """Calculate traditional and nonlinear betas for all stocks."""
        if not self.results or self.market_data is None:
            print("No data available. Run fetch_data() first.")
            return
            
        print("\nCalculating beta relationships...")
        
        # Align all data to common dates
        common_dates = self.market_data.index
        for symbol in list(self.results.keys()):
            self.results[symbol] = self.results[symbol].reindex(common_dates).dropna()
            if len(self.results[symbol]) < 100:  # Remove stocks with insufficient data
                del self.results[symbol]
                continue
            common_dates = common_dates.intersection(self.results[symbol].index)
        
        # Reindex all data to common dates
        self.market_data = self.market_data.reindex(common_dates)
        for symbol in self.results:
            self.results[symbol] = self.results[symbol].reindex(common_dates)
        
        print(f"Returns calculated for {len(common_dates)} common trading days")
        
        # Calculate returns
        market_returns = self.market_data['close'].pct_change().dropna()
        
        beta_results = {}
        
        for symbol, data in self.results.items():
            if len(data) < 100:
                continue
                
            stock_returns = data['close'].pct_change().dropna()
            
            # Align returns
            aligned_data = pd.DataFrame({
                'market': market_returns,
                'stock': stock_returns
            }).dropna()
            
            if len(aligned_data) < 100:
                continue
                
            # Split by positive/negative market days
            positive_days = aligned_data[aligned_data['market'] > 0]
            negative_days = aligned_data[aligned_data['market'] < 0]
            
            # Calculate betas
            if len(positive_days) > 20:
                pos_beta = np.cov(positive_days['stock'], positive_days['market'])[0,1] / np.var(positive_days['market'])
            else:
                pos_beta = np.nan
                
            if len(negative_days) > 20:
                neg_beta = np.cov(negative_days['stock'], negative_days['market'])[0,1] / np.var(negative_days['market'])
            else:
                neg_beta = np.nan
                
            # Traditional beta
            trad_beta = np.cov(aligned_data['stock'], aligned_data['market'])[0,1] / np.var(aligned_data['market'])
            
            # Beta ratio (positive/negative)
            if neg_beta != 0 and not np.isnan(neg_beta) and not np.isnan(pos_beta):
                beta_ratio = pos_beta / neg_beta
            else:
                beta_ratio = np.nan
                
            beta_results[symbol] = {
                'traditional_beta': trad_beta,
                'positive_beta': pos_beta,
                'negative_beta': neg_beta,
                'beta_ratio': beta_ratio,
                'positive_days': len(positive_days),
                'negative_days': len(negative_days)
            }
            
            print(f"{symbol}: Trad β={trad_beta:.3f}, Pos β={pos_beta:.3f}, Neg β={neg_beta:.3f}, Ratio={beta_ratio:.3f}")
        
        return beta_results
    
    def create_heatmap(self, beta_results, market_caps, top_n=50):
        """Create a true heatmap of beta ratios with squares sized by market cap."""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(beta_results, orient='index')
        
        # Add market cap
        df['market_cap'] = df.index.map(market_caps)
        
        # Filter for valid beta ratios
        df = df.dropna(subset=['beta_ratio'])
        
        # Sort by beta ratio (highest first)
        df = df.sort_values('beta_ratio', ascending=False).head(top_n)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Calculate grid dimensions
        n_companies = len(df)
        grid_size = int(np.ceil(np.sqrt(n_companies)))
        
        # Create a grid layout
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append((i, j))
        
        # Normalize market caps for sizing
        max_market_cap = df['market_cap'].max()
        min_market_cap = df['market_cap'].min()
        market_cap_range = max_market_cap - min_market_cap
        
        # Create squares for each company
        for idx, (symbol, row) in enumerate(df.iterrows()):
            if idx >= len(positions):
                break
                
            i, j = positions[idx]
            ratio = row['beta_ratio']
            market_cap = row['market_cap']
            
            # Size based on market cap (normalized)
            size_factor = 0.3 + 0.7 * (market_cap - min_market_cap) / market_cap_range
            square_size = size_factor
            
            # Color based on beta ratio
            if ratio > 1.1:
                color = 'red'  # High upside bias
            elif ratio > 1.0:
                color = 'orange'  # Moderate upside bias
            elif ratio > 0.9:
                color = 'yellow'  # Slight upside bias
            elif ratio > 0.8:
                color = 'lightblue'  # Slight downside bias
            else:
                color = 'blue'  # High downside bias
            
            # Create rectangle
            rect = plt.Rectangle((j - square_size/2, i - square_size/2), 
                               square_size, square_size, 
                               facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add text (ticker and ratio)
            ax.text(j, i, f'{symbol}\n{ratio:.3f}', 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='black' if ratio > 0.9 else 'white')
        
        # Set up the plot
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add title
        ax.set_title(f'S&P 500 Beta Asymmetry Heatmap\n(Top {len(df)} Companies by Positive/Negative Beta Ratio)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Ratio > 1.1 (High Upside)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='black', label='Ratio 1.0-1.1 (Moderate Upside)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='yellow', edgecolor='black', label='Ratio 0.9-1.0 (Slight Upside)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Ratio 0.8-0.9 (Slight Downside)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', label='Ratio < 0.8 (High Downside)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add summary statistics
        mean_ratio = df['beta_ratio'].mean()
        median_ratio = df['beta_ratio'].median()
        
        stats_text = f'Mean Ratio: {mean_ratio:.3f}\nMedian Ratio: {median_ratio:.3f}\n'
        stats_text += f'Companies with Ratio > 1.0: {(df["beta_ratio"] > 1.0).sum()}/{len(df)}\n'
        stats_text += f'Square size represents market cap\n'
        stats_text += f'Color represents beta ratio'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('docs/sp500_beta_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\nHeatmap saved to docs/sp500_beta_heatmap.png")
        print(f"Top 10 companies by beta ratio:")
        for i, symbol in enumerate(df.head(10).index):
            ratio = df.loc[symbol, 'beta_ratio']
            market_cap = df.loc[symbol, 'market_cap']
            print(f"  {i+1}. {symbol}: {ratio:.3f} (Market Cap: ${market_cap}B)")
        
        return df

def main():
    """Main function to run the S&P 500 beta heatmap analysis."""
    analyzer = SP500BetaAnalyzer()
    
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    market_caps = get_market_cap_data(symbols)
    
    # Fetch data
    if not analyzer.fetch_data(symbols):
        print("Failed to fetch data")
        return
    
    # Calculate betas
    beta_results = analyzer.calculate_betas()
    
    if not beta_results:
        print("No valid beta results")
        return
    
    # Create heatmap
    df = analyzer.create_heatmap(beta_results, market_caps, top_n=50)
    
    # Save results
    df.to_csv('sp500_beta_results.csv')
    print(f"\nResults saved to 'sp500_beta_results.csv'")
    
    return analyzer

if __name__ == "__main__":
    main() 