# Nonlinear Beta Analysis: Challenging CAPM Assumptions

This repository implements a comprehensive analysis of nonlinear beta relationships in stock returns, challenging the traditional CAPM assumption that stocks have constant sensitivity to market movements regardless of market direction.

## Research Question

**Do stocks behave differently in positive vs negative market environments?**

Traditional CAPM assumes that a stock's beta (sensitivity to market movements) is constant. This analysis tests whether stocks exhibit asymmetric behavior - potentially having different betas in up markets vs down markets.

## Key Findings

### Statistical Results (2021-2025 Analysis)
- **Sample Size**: 45 stocks, 1,139 trading days
- **Positive Market Days**: 620 (54.4%)
- **Negative Market Days**: 517 (45.6%)
- **Mean Traditional Beta**: 0.976
- **Mean Positive Market Beta**: 0.978
- **Mean Negative Market Beta**: 0.961
- **Mean Asymmetry Ratio**: 1.009

### Most Asymmetric Stocks
1. **PFE (Pfizer)**: 1.548 ratio - much higher downside beta
2. **LLY (Eli Lilly)**: 1.313 ratio - higher downside beta
3. **BMY (Bristol-Myers)**: 1.294 ratio - higher downside beta
4. **AMZN (Amazon)**: 1.235 ratio - higher downside beta
5. **RTX (Raytheon)**: 1.219 ratio - higher downside beta

## Quick Start

### Prerequisites
- Python 3.8+
- Alpaca Markets API account

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/chungderson/nonlinear-beta.git
cd nonlinear-beta
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Alpaca API**
Create a `config.json` file:
```json
{
    "ALPACA_KEY": "your_api_key_here",
    "ALPACA_SECRET": "your_secret_key_here"
}
```

### Run Analysis

```bash
python alpaca_nonlinear_beta_analysis_fixed.py
```

## Repository Structure

```
nonlinear-beta/
├── alpaca_nonlinear_beta_analysis_fixed.py  # Main analysis script
├── requirements.txt                          # Python dependencies
├── config.json                              # API configuration (not in repo)
├── alpaca_nonlinear_beta_results.csv        # Analysis results
├── README.md                               # This file
├── .gitignore                              # Git ignore rules
└── venv/                                   # Virtual environment (not in repo)
```

## Methodology

### Data Collection
- **Market Proxy**: SPY (S&P 500 ETF)
- **Stock Universe**: 45 major US stocks
- **Time Period**: 2021-2025 (1,139 trading days)
- **Data Source**: Alpaca Markets API (daily bars)

### Beta Calculations

1. **Traditional Beta** (CAPM):
   ```
   β_traditional = Cov(stock_returns, market_returns) / Var(market_returns)
   ```

2. **Positive Market Beta**:
   ```
   β_positive = Cov(stock_returns|market>0, market_returns|market>0) / Var(market_returns|market>0)
   ```

3. **Negative Market Beta**:
   ```
   β_negative = Cov(stock_returns|market<0, market_returns|market<0) / Var(market_returns|market<0)
   ```

4. **Asymmetry Ratio**:
   ```
   Asymmetry = β_negative / β_positive
   ```

### Statistical Testing
- **Paired T-Test**: Compares positive vs negative betas across all stocks
- **Confidence Intervals**: 95% confidence intervals for mean differences
- **Significance Levels**: *** p<0.001, ** p<0.01, * p<0.05

## Generated Visualizations

The analysis produces several key visualizations to help understand nonlinear beta relationships:

### 1. Beta Comparison Plots
![Beta Comparison](https://github.com/chungderson/nonlinear-beta/raw/main/docs/beta_comparison.png)

**Top Left**: Positive vs Negative Beta Scatter Plot
- Direct comparison of how stocks behave in up vs down markets
- Points above the diagonal line indicate higher sensitivity in down markets
- Points below the diagonal line indicate higher sensitivity in up markets

**Top Right**: Traditional vs Average Nonlinear Beta
- Compares traditional CAPM beta to the average of positive and negative betas
- Points on the diagonal line suggest traditional beta is a good approximation
- Deviations indicate nonlinear behavior

**Bottom Left**: Asymmetry Ratio Distribution
- Shows the ratio of negative to positive betas for each stock
- Ratio > 1: More sensitive in down markets
- Ratio < 1: More sensitive in up markets

**Bottom Right**: Beta Difference Analysis
- Absolute difference between negative and positive betas
- Larger bars indicate more asymmetric behavior

### 2. T-Test Results Visualization
![T-Test Results](https://github.com/chungderson/nonlinear-beta/raw/main/docs/t_test_results.png)

**Top Left**: Positive vs Negative Beta Scatter
- Each point represents a stock
- Red dashed line shows equal betas
- Green line shows mean difference

**Top Right**: Distribution of Beta Differences
- Histogram showing the spread of differences
- Red line at zero (no difference)
- Green line shows mean difference

**Bottom Left**: Box Plot Comparison
- Side-by-side comparison of positive vs negative betas
- Shows median, quartiles, and outliers

**Bottom Right**: Top 10 Stocks by Beta Difference
- Horizontal bar chart of largest differences
- Red bars: Higher negative beta
- Blue bars: Higher positive beta

### 3. Interactive Scatter Plots
The analysis also generates interactive Plotly visualizations that allow users to:
- Hover over points to see stock details
- Zoom and pan through the data
- Export high-quality images
- Explore relationships between different metrics

## Interpretation Guide

### Asymmetry Ratio
- **Ratio > 1**: Stock is more sensitive in down markets
- **Ratio < 1**: Stock is more sensitive in up markets
- **Ratio = 1**: Equal sensitivity (traditional CAPM holds)

### Investment Implications
- **High Asymmetry**: Traditional beta may underestimate risk
- **Downside Asymmetry**: Stock falls more in bad markets than expected
- **Upside Asymmetry**: Stock rises more in good markets than expected

## Results Summary

### Statistical Significance
- **T-Statistic**: Calculated from paired t-test comparing positive vs negative betas
- **P-Value**: Statistical significance of the difference
- **95% Confidence Interval**: Range where true mean difference likely falls

### Key Insights
1. **Nonlinear Behavior**: Many stocks show significant asymmetry
2. **Risk Management**: Traditional beta underestimates downside risk for asymmetric stocks
3. **Portfolio Construction**: Consider separate up/down betas for better risk modeling

## Customization

### Modify Stock Universe
Edit the `stock_symbols` list in `main()`:
```python
stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    # Add your preferred stocks here
]
```

### Change Time Period
Modify the `start_date` parameter:
```python
analyzer.fetch_data(stock_symbols, start_date='2020-01-01T00:00:00-04:00')
```

### Adjust Statistical Parameters
- Minimum data points for beta calculation
- Confidence level for t-tests
- Significance thresholds

## Dependencies

```
pandas>=2.0
matplotlib>=3.7
requests>=2.31
python-dateutil>=2.8
pytz>=2024.1
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
plotly>=5.15
seaborn>=0.12
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Alpaca Markets for providing market data
- Academic research on nonlinear beta relationships
- Open source financial analysis community

## Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: This analysis challenges traditional financial theory and should be used alongside other risk management tools. Past performance does not guarantee future results.

---

## Development Tools

This project was developed using:
- **Cursor IDE**: Advanced AI-powered code editor for rapid development and debugging
- **Claude Sonnet 4**: AI assistant for code generation, analysis, and documentation
- **GitHub**: Version control and repository hosting
- **Alpaca Markets API**: Financial data provider

*Built with modern AI-assisted development tools to accelerate research and analysis.* 