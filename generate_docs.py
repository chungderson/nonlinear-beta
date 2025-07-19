#!/usr/bin/env python3
"""
Generate documentation images for the README.
This script runs the analysis and saves the visualizations to the docs/ folder.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving

from alpaca_nonlinear_beta_analysis_fixed import NonlinearBetaAnalyzer

def main():
    """Generate documentation images."""
    
    print("Generating documentation images...")
    
    # Initialize analyzer
    analyzer = NonlinearBetaAnalyzer()
    
    # Use a smaller stock universe for faster generation
    stock_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'JPM', 'JNJ', 'PG', 'V', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
        'CRM', 'NKE'
    ]
    
    print(f"Analyzing {len(stock_symbols)} stocks for documentation...")
    
    # Fetch data (shorter period for faster generation)
    analyzer.fetch_data(stock_symbols, start_date='2023-01-01T00:00:00-04:00')
    
    # Calculate returns
    analyzer.calculate_returns()
    
    # Analyze all stocks
    results = analyzer.analyze_all_stocks()
    
    # Perform t-test
    t_test_results = analyzer.perform_t_test_on_betas()
    
    # Generate and save beta comparison plot
    print("Generating beta comparison plot...")
    analyzer.plot_beta_comparison(top_n=15)
    plt.savefig('docs/beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate and save t-test results plot
    print("Generating t-test results plot...")
    analyzer.plot_t_test_results(t_test_results)
    plt.savefig('docs/t_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Documentation images saved to docs/ folder!")
    print("Files generated:")
    print("- docs/beta_comparison.png")
    print("- docs/t_test_results.png")

if __name__ == "__main__":
    main() 