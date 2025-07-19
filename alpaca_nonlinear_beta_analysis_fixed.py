#!/usr/bin/env python3
"""
Nonlinear Beta Analysis using Alpaca Markets API
Analyzes beta relationships during positive vs negative market conditions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent windows from opening
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

# Import plotly for interactive plots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import your existing Alpaca functions
from getBars import getBars
from helperMethods import getTradingDays, calculateDrift

def load_config():
    """Load Alpaca configuration from config.json"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def getDailyBarAttributes(ticker, start_time=None, end_time=None):
    """
    Fetches daily bar attributes for a given stock ticker from the Alpaca API.
    Based on the user's proven Alpaca data fetching methods.
    
    Args:
        ticker (str): The stock ticker symbol to retrieve bar data for.
        start_time (datetime or str, optional): The start time for the bar data.
        end_time (datetime or str, optional): The end time for the bar data.
    
    Returns:
        pandas.DataFrame: DataFrame with columns [open, high, low, close, volume, timestamp]
    """
    config = load_config()
    
    url = "https://data.alpaca.markets/v2/stocks/bars"
    
    # Handle start_time
    if start_time is None:
        start_time = datetime.now() - timedelta(days=365)
    elif isinstance(start_time, str):
        time_part, offset = start_time.rsplit('-', 1)
        if ':' in offset:
            hours, minutes = map(int, offset.split(':'))
            offset_minutes = hours * 60 + minutes
        else:
            offset_minutes = int(offset) * 60
        start_time = datetime.fromisoformat(time_part)
        start_time = start_time + timedelta(minutes=offset_minutes)

    # Handle end_time
    if end_time is None:
        end_time = datetime.now()
    elif isinstance(end_time, str):
        time_part, offset = end_time.rsplit('-', 1)
        if ':' in offset:
            hours, minutes = map(int, offset.split(':'))
            offset_minutes = hours * 60 + minutes
        else:
            offset_minutes = int(offset) * 60
        end_time = datetime.fromisoformat(time_part)
        end_time = end_time + timedelta(minutes=offset_minutes)
    
    params = {
        "symbols": ticker,
        "timeframe": "1Day",
        "start": start_time.strftime("%Y-%m-%dT%H:%M:00Z"),
        "end": end_time.strftime("%Y-%m-%dT%H:%M:00Z"),
        "limit": 10000,
        "adjustment": "all",
        "feed": "sip",
        "sort": "asc"
    }
    
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": config['ALPACA_KEY'],
        "APCA-API-SECRET-KEY": config['ALPACA_SECRET']
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    data = response.json()
    
    if not data.get('bars') or not data['bars'].get(ticker) or len(data['bars'][ticker]) == 0:
        raise Exception(f"No bar data available for {ticker}")
    
    # Convert to DataFrame
    bars = data['bars'][ticker]
    df = pd.DataFrame(bars)
    
    # Rename columns to match previous format
    df = df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        't': 'timestamp'
    })
    
    # Reorder columns to match previous format
    df = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    
    return df

class NonlinearBetaAnalyzer:
    """
    Analyzer for nonlinear beta relationships in stock returns using Alpaca data.
    Compares traditional beta with separate betas for positive/negative market days.
    """
    
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        self.stock_data = {}
        self.market_data = None
        self.results = {}
        
    def fetch_data(self, stock_symbols, start_date='2021-01-01T00:00:00-04:00', end_date=None):
        """
        Fetch stock and market data for analysis.
        
        Parameters:
        - stock_symbols: List of stock symbols to analyze
        - start_date: Start date for analysis
        - end_date: End date for analysis (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S-04:00')
        
        print(f"Fetching data for {len(stock_symbols)} stocks and market index...")
        
        # Fetch market data (SPY as proxy)
        print("Fetching market data (SPY)...")
        try:
            self.market_data = getDailyBarAttributes('SPY', start_date, end_date)
            print(f"✓ Market data: {len(self.market_data)} days")
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return
        
        # Fetch stock data
        for symbol in stock_symbols:
            try:
                print(f"Fetching data for {symbol}...")
                stock_data = getDailyBarAttributes(symbol, start_date, end_date)
                
                if not stock_data.empty:
                    self.stock_data[symbol] = stock_data
                    print(f"✓ {symbol}: {len(stock_data)} days of data")
                else:
                    print(f"✗ {symbol}: No data available")
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"✗ {symbol}: Error fetching data - {e}")
        
        print(f"Data collection complete. Market: {len(self.market_data)} days")
        
    def calculate_returns(self):
        """Calculate daily returns for all stocks and market."""
        # Convert timestamp to datetime index for market data
        self.market_data['timestamp'] = pd.to_datetime(self.market_data['timestamp'])
        self.market_data.set_index('timestamp', inplace=True)
        
        # Market returns
        self.market_returns = self.market_data['close'].pct_change().dropna()
        
        # Stock returns
        self.stock_returns = {}
        for symbol, prices in self.stock_data.items():
            # Convert timestamp to datetime index for stock data
            prices['timestamp'] = pd.to_datetime(prices['timestamp'])
            prices.set_index('timestamp', inplace=True)
            self.stock_returns[symbol] = prices['close'].pct_change().dropna()
            
        # Align dates
        common_dates = self.market_returns.index.intersection(
            pd.concat(self.stock_returns.values(), axis=1).index
        )
        
        self.market_returns = self.market_returns.loc[common_dates]
        for symbol in self.stock_returns:
            self.stock_returns[symbol] = self.stock_returns[symbol].loc[common_dates]
            
        print(f"Returns calculated for {len(common_dates)} common trading days")
        
    def calculate_traditional_beta(self, stock_returns, market_returns):
        """Calculate traditional (linear) beta."""
        # Remove any NaN values
        valid_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(valid_data) < 30:  # Need minimum data points
            return np.nan, np.nan
            
        # Calculate beta using covariance method
        covariance = np.cov(valid_data['stock'], valid_data['market'])[0, 1]
        market_variance = np.var(valid_data['market'])
        
        if market_variance == 0:
            return np.nan, np.nan
            
        beta = covariance / market_variance
        
        # Calculate R-squared
        correlation = np.corrcoef(valid_data['stock'], valid_data['market'])[0, 1]
        r_squared = correlation ** 2
        
        return beta, r_squared
        
    def calculate_nonlinear_beta(self, stock_returns, market_returns):
        """
        Calculate separate betas for positive and negative market days.
        
        Returns:
        - beta_positive: Beta when market is positive
        - beta_negative: Beta when market is negative
        - asymmetry_ratio: beta_negative / beta_positive
        """
        # Remove any NaN values
        valid_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(valid_data) < 30:
            return np.nan, np.nan, np.nan
            
        # Separate positive and negative market days
        positive_days = valid_data[valid_data['market'] > 0]
        negative_days = valid_data[valid_data['market'] < 0]
        
        # Calculate betas for each regime
        beta_positive = np.nan
        beta_negative = np.nan
        
        if len(positive_days) >= 10:
            covariance_pos = np.cov(positive_days['stock'], positive_days['market'])[0, 1]
            market_var_pos = np.var(positive_days['market'])
            if market_var_pos > 0:
                beta_positive = covariance_pos / market_var_pos
                
        if len(negative_days) >= 10:
            covariance_neg = np.cov(negative_days['stock'], negative_days['market'])[0, 1]
            market_var_neg = np.var(negative_days['market'])
            if market_var_neg > 0:
                beta_negative = covariance_neg / market_var_neg
                
        # Calculate asymmetry ratio
        asymmetry_ratio = np.nan
        if not np.isnan(beta_positive) and not np.isnan(beta_negative) and beta_positive != 0:
            asymmetry_ratio = beta_negative / beta_positive
            
        return beta_positive, beta_negative, asymmetry_ratio
        
    def analyze_all_stocks(self):
        """Analyze all stocks for traditional and nonlinear beta."""
        print("\nAnalyzing beta relationships...")
        
        results = {}
        
        for symbol in self.stock_returns:
            stock_ret = self.stock_returns[symbol]
            market_ret = self.market_returns
            
            # Traditional beta
            traditional_beta, r_squared = self.calculate_traditional_beta(stock_ret, market_ret)
            
            # Nonlinear beta
            beta_pos, beta_neg, asymmetry = self.calculate_nonlinear_beta(stock_ret, market_ret)
            
            # Additional statistics
            positive_days = len(market_ret[market_ret > 0])
            negative_days = len(market_ret[market_ret < 0])
            
            results[symbol] = {
                'traditional_beta': traditional_beta,
                'r_squared': r_squared,
                'beta_positive': beta_pos,
                'beta_negative': beta_neg,
                'asymmetry_ratio': asymmetry,
                'positive_days': positive_days,
                'negative_days': negative_days,
                'total_days': len(stock_ret)
            }
            
            print(f"{symbol}: Traditional β={traditional_beta:.3f}, "
                  f"Positive β={beta_pos:.3f}, Negative β={beta_neg:.3f}, "
                  f"Asymmetry={asymmetry:.3f}")
        
        self.results = results
        return results
        
    def perform_t_test_on_betas(self):
        """Perform paired t-test on positive vs negative betas."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return None
            
        # Extract valid beta pairs
        beta_pairs = []
        for symbol, result in self.results.items():
            if (not np.isnan(result['beta_positive']) and 
                not np.isnan(result['beta_negative']) and
                result['beta_positive'] != 0):
                beta_pairs.append({
                    'symbol': symbol,
                    'beta_positive': result['beta_positive'],
                    'beta_negative': result['beta_negative'],
                    'difference': result['beta_positive'] - result['beta_negative']
                })
        
        if len(beta_pairs) < 2:
            print("Insufficient data for t-test")
            return None
            
        # Convert to DataFrame for easier analysis
        df_pairs = pd.DataFrame(beta_pairs)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(df_pairs['beta_positive'], df_pairs['beta_negative'])
        
        # Calculate additional statistics
        mean_diff = df_pairs['difference'].mean()
        std_diff = df_pairs['difference'].std()
        n_stocks = len(df_pairs)
        
        # Calculate confidence interval
        confidence_level = 0.95
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n_stocks - 1)
        margin_of_error = t_critical * (std_diff / np.sqrt(n_stocks))
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error
        
        # Print results
        print("\n" + "="*60)
        print("PAIRED T-TEST: POSITIVE vs NEGATIVE BETA")
        print("="*60)
        print(f"Sample Size: {n_stocks} stocks")
        print(f"Mean Positive Beta: {df_pairs['beta_positive'].mean():.4f}")
        print(f"Mean Negative Beta: {df_pairs['beta_negative'].mean():.4f}")
        print(f"Mean Difference (Positive - Negative): {mean_diff:.4f}")
        print(f"Standard Deviation of Differences: {std_diff:.4f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Interpret results
        print(f"\nInterpretation:")
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "Not significant (p >= 0.05)"
            
        print(f"  The difference between positive and negative betas is {significance}")
        
        if p_value < 0.05:
            if mean_diff > 0:
                print(f"  Positive market betas are significantly HIGHER than negative market betas")
            else:
                print(f"  Positive market betas are significantly LOWER than negative market betas")
        else:
            print(f"  No significant difference between positive and negative market betas")
            
        # Show stocks with significant individual differences
        print(f"\nStocks with largest beta differences:")
        df_sorted = df_pairs.sort_values('difference', key=abs, ascending=False)
        for _, row in df_sorted.head(10).iterrows():
            direction = "higher" if row['difference'] > 0 else "lower"
            print(f"  {row['symbol']}: {row['difference']:.4f} ({direction} positive beta)")
            
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'n_stocks': n_stocks,
            'significant': p_value < 0.05,
            'df_pairs': df_pairs
        }
        
    def create_summary_dataframe(self):
        """Create a summary DataFrame of all results."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return None
            
        df = pd.DataFrame(self.results).T
        
        # Add some derived metrics
        df['beta_difference'] = df['beta_positive'] - df['beta_negative']
        df['abs_asymmetry'] = abs(df['asymmetry_ratio'])
        df['volatility'] = [self.stock_returns[symbol].std() for symbol in df.index]
        df['avg_return'] = [self.stock_returns[symbol].mean() for symbol in df.index]
        
        return df
        
    def plot_beta_comparison(self, top_n=20, save_path=None):
        """Plot comparison of positive vs negative betas and asymmetry analysis."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return
            
        df = self.create_summary_dataframe()
        df = df.sort_values('traditional_beta', ascending=False).head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Positive vs Negative Beta (more meaningful comparison)
        axes[0, 0].scatter(df['beta_positive'], df['beta_negative'], alpha=0.7)
        axes[0, 0].plot([0, 2.5], [0, 2.5], 'r--', alpha=0.5, label='Equal betas')
        axes[0, 0].set_xlabel('Positive Market Beta')
        axes[0, 0].set_ylabel('Negative Market Beta')
        axes[0, 0].set_title('Positive vs Negative Market Beta')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Traditional Beta vs Average of Positive and Negative
        df['avg_nonlinear_beta'] = (df['beta_positive'] + df['beta_negative']) / 2
        axes[0, 1].scatter(df['traditional_beta'], df['avg_nonlinear_beta'], alpha=0.7)
        axes[0, 1].plot([0, 2.5], [0, 2.5], 'r--', alpha=0.5, label='Equal betas')
        axes[0, 1].set_xlabel('Traditional Beta')
        axes[0, 1].set_ylabel('Average Nonlinear Beta')
        axes[0, 1].set_title('Traditional vs Average Nonlinear Beta')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Asymmetry Ratio
        axes[1, 0].bar(range(len(df)), df['asymmetry_ratio'], alpha=0.7)
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Stock Index')
        axes[1, 0].set_ylabel('Asymmetry Ratio (β_neg / β_pos)')
        axes[1, 0].set_title('Beta Asymmetry Ratio')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Beta Difference
        axes[1, 1].bar(range(len(df)), df['beta_difference'], alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Stock Index')
        axes[1, 1].set_ylabel('Beta Difference (β_pos - β_neg)')
        axes[1, 1].set_title('Beta Difference')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df.index, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close the specific figure
        else:
            plt.show()
        
    def plot_interactive_scatter(self):
        """Create an interactive scatter plot using Plotly."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return
            
        df = self.create_summary_dataframe()
        
    def plot_t_test_results(self, t_test_results, save_path=None):
        """Plot t-test results showing the distribution of beta differences."""
        if not t_test_results or 'df_pairs' not in t_test_results:
            print("No t-test results available.")
            return
            
        df_pairs = t_test_results['df_pairs']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot of positive vs negative betas
        axes[0, 0].scatter(df_pairs['beta_positive'], df_pairs['beta_negative'], alpha=0.7)
        axes[0, 0].plot([0, 2.5], [0, 2.5], 'r--', alpha=0.5, label='Equal betas')
        axes[0, 0].set_xlabel('Positive Market Beta')
        axes[0, 0].set_ylabel('Negative Market Beta')
        axes[0, 0].set_title('Positive vs Negative Market Betas')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Histogram of beta differences
        axes[0, 1].hist(df_pairs['difference'], bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='No difference')
        axes[0, 1].axvline(x=t_test_results['mean_difference'], color='g', linestyle='-', 
                           alpha=0.7, label=f'Mean diff: {t_test_results["mean_difference"]:.4f}')
        axes[0, 1].set_xlabel('Beta Difference (Positive - Negative)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Beta Differences')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot of positive vs negative betas
        beta_data = [df_pairs['beta_positive'], df_pairs['beta_negative']]
        axes[1, 0].boxplot(beta_data, labels=['Positive Beta', 'Negative Beta'])
        axes[1, 0].set_ylabel('Beta Value')
        axes[1, 0].set_title('Box Plot: Positive vs Negative Betas')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Bar plot of top differences
        top_differences = df_pairs.sort_values('difference', key=abs, ascending=False).head(10)
        colors = ['red' if x < 0 else 'blue' for x in top_differences['difference']]
        axes[1, 1].barh(range(len(top_differences)), top_differences['difference'], color=colors, alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_differences)))
        axes[1, 1].set_yticklabels(top_differences['symbol'])
        axes[1, 1].set_xlabel('Beta Difference (Positive - Negative)')
        axes[1, 1].set_title('Top 10 Stocks by Beta Difference')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close the specific figure
        else:
            plt.show()
        
        # Print t-test summary
        print(f"\nT-Test Summary:")
        print(f"  P-value: {t_test_results['p_value']:.6f}")
        print(f"  Significant at 5% level: {t_test_results['significant']}")
        print(f"  Mean difference: {t_test_results['mean_difference']:.4f}")
        print(f"  95% CI: [{t_test_results['confidence_interval'][0]:.4f}, {t_test_results['confidence_interval'][1]:.4f}]")
        print(f"  Interpretation: Positive - Negative Beta Difference")
        
    def plot_interactive_scatter(self):
        """Create an interactive scatter plot using Plotly."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return
            
        df = self.create_summary_dataframe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Positive vs Negative Beta', 'Traditional vs Average Nonlinear Beta',
                          'Asymmetry Ratio vs Traditional Beta', 'Volatility vs Asymmetry'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Positive vs Negative Beta
        fig.add_trace(
            go.Scatter(x=df['beta_positive'], y=df['beta_negative'],
                      mode='markers', name='Positive vs Negative',
                      text=df.index, hovertemplate='<b>%{text}</b><br>' +
                      'Positive Beta: %{x:.3f}<br>' +
                      'Negative Beta: %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
        
        # Traditional vs Average Nonlinear Beta
        df['avg_nonlinear_beta'] = (df['beta_positive'] + df['beta_negative']) / 2
        fig.add_trace(
            go.Scatter(x=df['traditional_beta'], y=df['avg_nonlinear_beta'],
                      mode='markers', name='Traditional vs Average',
                      text=df.index, hovertemplate='<b>%{text}</b><br>' +
                      'Traditional Beta: %{x:.3f}<br>' +
                      'Average Nonlinear Beta: %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        
        # Asymmetry vs Traditional Beta
        fig.add_trace(
            go.Scatter(x=df['traditional_beta'], y=df['asymmetry_ratio'],
                      mode='markers', name='Asymmetry',
                      text=df.index, hovertemplate='<b>%{text}</b><br>' +
                      'Traditional Beta: %{x:.3f}<br>' +
                      'Asymmetry: %{y:.3f}<extra></extra>'),
            row=2, col=1
        )
        
        # Volatility vs Asymmetry
        fig.add_trace(
            go.Scatter(x=df['volatility'], y=df['asymmetry_ratio'],
                      mode='markers', name='Volatility vs Asymmetry',
                      text=df.index, hovertemplate='<b>%{text}</b><br>' +
                      'Volatility: %{x:.3f}<br>' +
                      'Asymmetry: %{y:.3f}<extra></extra>'),
            row=2, col=2
        )
        
        # Add reference lines
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Nonlinear Beta Analysis (Alpaca Data)")
        fig.show()
        
    def generate_report(self, t_test_results=None):
        """Generate a comprehensive analysis report."""
        if not self.results:
            print("No results available. Run analyze_all_stocks() first.")
            return
            
        df = self.create_summary_dataframe()
        
        print("\n" + "="*60)
        print("NONLINEAR BETA ANALYSIS REPORT (ALPACA DATA)")
        print("="*60)
        
        # Summary statistics
        print(f"\nAnalysis Period: {self.market_returns.index[0].date()} to {self.market_returns.index[-1].date()}")
        print(f"Total Trading Days: {len(self.market_returns)}")
        print(f"Positive Market Days: {len(self.market_returns[self.market_returns > 0])}")
        print(f"Negative Market Days: {len(self.market_returns[self.market_returns < 0])}")
        
        # Traditional beta statistics
        valid_traditional = df['traditional_beta'].dropna()
        print(f"\nTraditional Beta Statistics:")
        print(f"  Mean: {valid_traditional.mean():.3f}")
        print(f"  Median: {valid_traditional.median():.3f}")
        print(f"  Std Dev: {valid_traditional.std():.3f}")
        print(f"  Range: {valid_traditional.min():.3f} to {valid_traditional.max():.3f}")
        
        # Nonlinear beta statistics
        valid_pos = df['beta_positive'].dropna()
        valid_neg = df['beta_negative'].dropna()
        valid_asym = df['asymmetry_ratio'].dropna()
        
        print(f"\nPositive Market Beta Statistics:")
        print(f"  Mean: {valid_pos.mean():.3f}")
        print(f"  Median: {valid_pos.median():.3f}")
        print(f"  Std Dev: {valid_pos.std():.3f}")
        
        print(f"\nNegative Market Beta Statistics:")
        print(f"  Mean: {valid_neg.mean():.3f}")
        print(f"  Median: {valid_neg.median():.3f}")
        print(f"  Std Dev: {valid_neg.std():.3f}")
        
        print(f"\nAsymmetry Ratio Statistics:")
        print(f"  Mean: {valid_asym.mean():.3f}")
        print(f"  Median: {valid_asym.median():.3f}")
        print(f"  Stocks with β_neg > β_pos: {len(valid_asym[valid_asym > 1])}")
        print(f"  Stocks with β_neg < β_pos: {len(valid_asym[valid_asym < 1])}")
        
        # T-Test Results
        if t_test_results:
            print(f"\n" + "="*40)
            print("STATISTICAL T-TEST RESULTS")
            print("="*40)
            print(f"Sample Size: {t_test_results['n_stocks']} stocks")
            print(f"Mean Positive Beta: {t_test_results['df_pairs']['beta_positive'].mean():.4f}")
            print(f"Mean Negative Beta: {t_test_results['df_pairs']['beta_negative'].mean():.4f}")
            print(f"Mean Difference (Negative - Positive): {t_test_results['mean_difference']:.4f}")
            print(f"Standard Deviation of Differences: {t_test_results['std_difference']:.4f}")
            print(f"T-statistic: {t_test_results['t_statistic']:.4f}")
            print(f"P-value: {t_test_results['p_value']:.6f}")
            print(f"95% Confidence Interval: [{t_test_results['confidence_interval'][0]:.4f}, {t_test_results['confidence_interval'][1]:.4f}]")
            
            # Significance interpretation
            if t_test_results['p_value'] < 0.001:
                significance = "*** (p < 0.001)"
            elif t_test_results['p_value'] < 0.01:
                significance = "** (p < 0.01)"
            elif t_test_results['p_value'] < 0.05:
                significance = "* (p < 0.05)"
            else:
                significance = "Not significant (p >= 0.05)"
                
            print(f"\nStatistical Significance: {significance}")
            
            if t_test_results['significant']:
                if t_test_results['mean_difference'] > 0:
                    print(f"  → Negative market betas are significantly HIGHER than positive market betas")
                else:
                    print(f"  → Negative market betas are significantly LOWER than positive market betas")
            else:
                print(f"  → No significant difference between positive and negative market betas")
                
            # Top differences
            print(f"\nStocks with Largest Beta Differences:")
            df_sorted = t_test_results['df_pairs'].sort_values('difference', key=abs, ascending=False)
            for _, row in df_sorted.head(5).iterrows():
                direction = "higher" if row['difference'] > 0 else "lower"
                print(f"  {row['symbol']}: {row['difference']:.4f} ({direction} negative beta)")
        
        # Top and bottom performers
        print(f"\nTop 5 Stocks by Traditional Beta:")
        top_traditional = df.nlargest(5, 'traditional_beta')[['traditional_beta', 'beta_positive', 'beta_negative', 'asymmetry_ratio']]
        print(top_traditional.round(3))
        
        print(f"\nTop 5 Stocks by Asymmetry Ratio:")
        top_asymmetry = df.nlargest(5, 'asymmetry_ratio')[['traditional_beta', 'beta_positive', 'beta_negative', 'asymmetry_ratio']]
        print(top_asymmetry.round(3))
        
        print(f"\nBottom 5 Stocks by Asymmetry Ratio:")
        bottom_asymmetry = df.nsmallest(5, 'asymmetry_ratio')[['traditional_beta', 'beta_positive', 'beta_negative', 'asymmetry_ratio']]
        print(bottom_asymmetry.round(3))
        
        print("\n" + "="*60)

def main():
    """Main function to demonstrate the nonlinear beta analysis with Alpaca data."""
    
    # Example stock symbols (you can modify this list)
    stock_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'JPM', 'JNJ', 'PG', 'V', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
        'CRM', 'NKE', 'INTC', 'VZ', 'CMCSA', 'PFE', 'TMO', 'ABT', 'KO',
        'PEP', 'TXN', 'MRK', 'AVGO', 'QCOM', 'COST', 'ACN', 'DHR', 'LLY',
        'BMY', 'UNP', 'RTX', 'LOW', 'SPGI', 'ISRG', 'INTU', 'ADI', 'BDX'
    ]
    
    # Initialize analyzer
    analyzer = NonlinearBetaAnalyzer()
    
    # Fetch data (last 3 years)
    analyzer.fetch_data(stock_symbols, start_date='2021-01-01T00:00:00-04:00')
    
    # Calculate returns
    analyzer.calculate_returns()
    
    # Analyze all stocks
    results = analyzer.analyze_all_stocks()
    
    # Perform t-test on beta differences
    t_test_results = analyzer.perform_t_test_on_betas()
    
    # Generate report
    analyzer.generate_report(t_test_results)
    
    # Create visualizations
    analyzer.plot_beta_comparison()
    analyzer.plot_interactive_scatter()
    
    # Plot t-test results
    if t_test_results:
        analyzer.plot_t_test_results(t_test_results)
    
    # Save main visualizations to docs folder
    print("\nSaving main analysis visualizations to docs/ folder...")
    
    # Generate and save beta comparison plot
    analyzer.plot_beta_comparison(save_path='docs/beta_comparison.png')
    
    # Generate and save t-test results plot
    if t_test_results:
        analyzer.plot_t_test_results(t_test_results, save_path='docs/t_test_results.png')
    
    print("Main analysis visualizations saved to docs/ folder!")
    
    # Save results to CSV
    df = analyzer.create_summary_dataframe()
    df.to_csv('alpaca_nonlinear_beta_results.csv')
    print(f"\nResults saved to 'alpaca_nonlinear_beta_results.csv'")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main() 