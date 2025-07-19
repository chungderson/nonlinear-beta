#!/usr/bin/env python3
"""
Example usage of the Nonlinear Beta Analysis framework.

This script demonstrates how to use the NonlinearBetaAnalyzer class
to perform nonlinear beta analysis on a small subset of stocks.
"""

from alpaca_nonlinear_beta_analysis_fixed import NonlinearBetaAnalyzer

def main():
    """Example analysis with a small stock universe."""
    
    print("=== Nonlinear Beta Analysis Example ===\n")
    
    # Initialize analyzer
    analyzer = NonlinearBetaAnalyzer()
    
    # Define a small stock universe for testing
    stock_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
        'JPM', 'JNJ', 'PG', 'V', 'UNH'             # Various sectors
    ]
    
    print(f"Analyzing {len(stock_symbols)} stocks...")
    print(f"Stocks: {', '.join(stock_symbols)}")
    
    # Fetch data (last 2 years for faster testing)
    analyzer.fetch_data(stock_symbols, start_date='2023-01-01T00:00:00-04:00')
    
    # Calculate returns
    analyzer.calculate_returns()
    
    # Analyze all stocks
    results = analyzer.analyze_all_stocks()
    
    # Perform t-test
    t_test_results = analyzer.perform_t_test_on_betas()
    
    # Generate report
    analyzer.generate_report(t_test_results)
    
    # Create visualizations
    analyzer.plot_beta_comparison(top_n=10)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated plots and CSV file for detailed results.")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main() 