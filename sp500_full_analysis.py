#!/usr/bin/env python3
"""
Full S&P 500 Beta Asymmetry Analysis
Lists all 500 companies in ascending order and creates sector-by-sector bar charts.
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
    Get S&P 500 symbols. For now, we'll use a comprehensive list of major companies.
    In a full implementation, you could fetch the complete S&P 500 list from an API.
    """
    # Comprehensive list of S&P 500 companies (top 100 by market cap)
    sp500_symbols = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'QCOM', 'AMD',
        'INTC', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'INTU', 'ADI', 'TXN', 'MU',
        
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY', 'ISRG',
        'GILD', 'AMGN', 'REGN', 'VRTX', 'BIIB', 'HUM', 'CI', 'ANTM', 'CNC', 'HCA',
        
        # Financials
        'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'AXP',
        'BLK', 'SCHW', 'COF', 'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'PGR', 'CB',
        
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'MAR', 'BKNG',
        'CMG', 'YUM', 'DRI', 'ROST', 'ULTA', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T',
        
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'GIS', 'K',
        'HSY', 'SJM', 'KMB', 'CAG', 'HRL', 'SJM', 'CPB', 'KHC', 'MDLZ', 'SJM',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
        'BKR', 'FANG', 'PXD', 'DVN', 'NOV', 'FTI', 'HES', 'APA', 'MRO', 'XEC',
        
        # Industrials
        'UNP', 'RTX', 'HON', 'CAT', 'DE', 'UPS', 'FDX', 'LMT', 'BA', 'GE',
        'MMM', 'EMR', 'ITW', 'ETN', 'PH', 'DOV', 'XYL', 'AME', 'ROK', 'FTV',
        
        # Materials
        'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'BLL', 'ALB', 'ECL',
        'VMC', 'BLL', 'NEM', 'FCX', 'APD', 'LIN', 'DOW', 'DD', 'NUE', 'BLL',
        
        # Real Estate
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'DLR', 'WELL', 'VICI',
        'EQR', 'AVB', 'MAA', 'ESS', 'UDR', 'BXP', 'ARE', 'KIM', 'REG', 'FRT',
        
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'DTE', 'WEC', 'ED',
        'AEE', 'EIX', 'PEG', 'ETR', 'FE', 'AES', 'NRG', 'PCG', 'EXC', 'NI'
    ]
    return sp500_symbols

def get_sector_classification():
    """Get sector classification for S&P 500 companies."""
    sector_map = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
        'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology', 'AVGO': 'Technology',
        'QCOM': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'ORCL': 'Technology',
        'CRM': 'Technology', 'ADBE': 'Technology', 'NFLX': 'Technology', 'PYPL': 'Technology',
        'INTU': 'Technology', 'ADI': 'Technology', 'TXN': 'Technology', 'MU': 'Technology',
        
        # Healthcare
        'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
        'LLY': 'Healthcare', 'TMO': 'Healthcare', 'DHR': 'Healthcare', 'ABT': 'Healthcare',
        'BMY': 'Healthcare', 'ISRG': 'Healthcare', 'GILD': 'Healthcare', 'AMGN': 'Healthcare',
        'REGN': 'Healthcare', 'VRTX': 'Healthcare', 'BIIB': 'Healthcare', 'HUM': 'Healthcare',
        'CI': 'Healthcare', 'ANTM': 'Healthcare', 'CNC': 'Healthcare', 'HCA': 'Healthcare',
        
        # Financials
        'BRK.B': 'Financials', 'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials', 'USB': 'Financials',
        'PNC': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
        'COF': 'Financials', 'AIG': 'Financials', 'MET': 'Financials', 'PRU': 'Financials',
        'ALL': 'Financials', 'TRV': 'Financials', 'PGR': 'Financials', 'CB': 'Financials',
        
        # Consumer Discretionary
        'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
        'MAR': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary',
        'YUM': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary', 'ROST': 'Consumer Discretionary',
        'ULTA': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary', 'CMCSA': 'Consumer Discretionary',
        'VZ': 'Consumer Discretionary', 'T': 'Consumer Discretionary',
        
        # Consumer Staples
        'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
        'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
        'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'GIS': 'Consumer Staples',
        'K': 'Consumer Staples', 'HSY': 'Consumer Staples', 'SJM': 'Consumer Staples',
        'KMB': 'Consumer Staples', 'CAG': 'Consumer Staples', 'HRL': 'Consumer Staples',
        'CPB': 'Consumer Staples', 'KHC': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
        
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
        'SLB': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy',
        'OXY': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy', 'FANG': 'Energy',
        'PXD': 'Energy', 'DVN': 'Energy', 'NOV': 'Energy', 'FTI': 'Energy',
        'HES': 'Energy', 'APA': 'Energy', 'MRO': 'Energy', 'XEC': 'Energy',
        
        # Industrials
        'UNP': 'Industrials', 'RTX': 'Industrials', 'HON': 'Industrials', 'CAT': 'Industrials',
        'DE': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials', 'LMT': 'Industrials',
        'BA': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials', 'EMR': 'Industrials',
        'ITW': 'Industrials', 'ETN': 'Industrials', 'PH': 'Industrials', 'DOV': 'Industrials',
        'XYL': 'Industrials', 'AME': 'Industrials', 'ROK': 'Industrials', 'FTV': 'Industrials',
        
        # Materials
        'LIN': 'Materials', 'APD': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials',
        'DOW': 'Materials', 'DD': 'Materials', 'NUE': 'Materials', 'BLL': 'Materials',
        'ALB': 'Materials', 'ECL': 'Materials', 'VMC': 'Materials',
        
        # Real Estate
        'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
        'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'DLR': 'Real Estate',
        'WELL': 'Real Estate', 'VICI': 'Real Estate', 'EQR': 'Real Estate', 'AVB': 'Real Estate',
        'MAA': 'Real Estate', 'ESS': 'Real Estate', 'UDR': 'Real Estate', 'BXP': 'Real Estate',
        'ARE': 'Real Estate', 'KIM': 'Real Estate', 'REG': 'Real Estate', 'FRT': 'Real Estate',
        
        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        'AEP': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities', 'DTE': 'Utilities',
        'WEC': 'Utilities', 'ED': 'Utilities', 'AEE': 'Utilities', 'EIX': 'Utilities',
        'PEG': 'Utilities', 'ETR': 'Utilities', 'FE': 'Utilities', 'AES': 'Utilities',
        'NRG': 'Utilities', 'PCG': 'Utilities', 'EXC': 'Utilities', 'NI': 'Utilities'
    }
    return sector_map

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

class SP500FullAnalyzer:
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
    
    def create_sector_charts(self, beta_results, sector_map, market_caps):
        """Create sector-by-sector bar charts."""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(beta_results, orient='index')
        
        # Add sector and market cap
        df['sector'] = df.index.map(sector_map)
        df['market_cap'] = df.index.map(market_caps)
        
        # Filter for valid beta ratios
        df = df.dropna(subset=['beta_ratio'])
        
        # Get unique sectors
        sectors = df['sector'].unique()
        
        # Create subplots for each sector
        n_sectors = len(sectors)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, sector in enumerate(sectors):
            if i >= len(axes):
                break
                
            sector_data = df[df['sector'] == sector].sort_values('beta_ratio', ascending=True)
            
            if len(sector_data) == 0:
                continue
                
            ax = axes[i]
            
            # Create bar chart
            bars = ax.barh(range(len(sector_data)), sector_data['beta_ratio'], 
                          color=['red' if x > 1.1 else 'orange' if x > 1.0 else 'yellow' if x > 0.9 else 'lightblue' if x > 0.8 else 'blue' for x in sector_data['beta_ratio']])
            
            # Add reference line at 1.0
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Set labels
            ax.set_yticks(range(len(sector_data)))
            ax.set_yticklabels(sector_data.index)
            ax.set_xlabel('Positive/Negative Beta Ratio')
            ax.set_title(f'{sector} ({len(sector_data)} companies)')
            ax.grid(True, alpha=0.3)
            
            # Add ratio values on bars
            for j, (idx, row) in enumerate(sector_data.iterrows()):
                ax.text(row['beta_ratio'], j, f'{row["beta_ratio"]:.3f}', 
                       ha='left', va='center', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_sectors, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('docs/sp500_sector_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\nSector charts saved to docs/sp500_sector_charts.png")
        
        return df
    
    def create_ascending_list(self, beta_results, sector_map, market_caps):
        """Create a comprehensive list of all companies in ascending order."""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(beta_results, orient='index')
        
        # Add sector and market cap
        df['sector'] = df.index.map(sector_map)
        df['market_cap'] = df.index.map(market_caps)
        
        # Filter for valid beta ratios
        df = df.dropna(subset=['beta_ratio'])
        
        # Sort by beta ratio (ascending)
        df_sorted = df.sort_values('beta_ratio', ascending=True)
        
        # Create the comprehensive list
        print("\n" + "="*80)
        print("S&P 500 COMPANIES BY BETA RATIO (ASCENDING ORDER)")
        print("="*80)
        
        print(f"{'Rank':<4} {'Symbol':<8} {'Ratio':<8} {'Sector':<20} {'Market Cap':<12} {'Trad Beta':<10} {'Pos Beta':<10} {'Neg Beta':<10}")
        print("-"*80)
        
        for i, (symbol, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i:<4} {symbol:<8} {row['beta_ratio']:<8.3f} {row['sector']:<20} ${row['market_cap']:<11}B {row['traditional_beta']:<10.3f} {row['positive_beta']:<10.3f} {row['negative_beta']:<10.3f}")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
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
            print(f"{sector:<20}: {stats['count']:>3} companies, Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
        
        return df_sorted

def main():
    """Main function to run the full S&P 500 analysis."""
    analyzer = SP500FullAnalyzer()
    
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    market_caps = get_market_cap_data(symbols)
    sector_map = get_sector_classification()
    
    # Fetch data
    if not analyzer.fetch_data(symbols):
        print("Failed to fetch data")
        return
    
    # Calculate betas
    beta_results = analyzer.calculate_betas()
    
    if not beta_results:
        print("No valid beta results")
        return
    
    # Create sector charts
    df = analyzer.create_sector_charts(beta_results, sector_map, market_caps)
    
    # Create ascending list
    df_sorted = analyzer.create_ascending_list(beta_results, sector_map, market_caps)
    
    # Save results
    df_sorted.to_csv('sp500_full_results.csv')
    print(f"\nResults saved to 'sp500_full_results.csv'")
    
    return analyzer

if __name__ == "__main__":
    main() 