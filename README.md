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

## S&P 500 Beta Asymmetry Heatmap Analysis

### Market Cap Weighted Analysis
A comprehensive analysis of the top 50 S&P 500 companies by market capitalization, examining their positive-to-negative beta ratios to identify asymmetric behavior patterns.

![S&P 500 Beta Heatmap](https://github.com/chungderson/nonlinear-beta/raw/main/docs/sp500_beta_heatmap.png)

### Key S&P 500 Findings

**Top 10 Companies by Beta Ratio (Positive/Negative):**
1. **UNH** (UnitedHealth): 1.299 ratio - $500B market cap
2. **TXN** (Texas Instruments): 1.255 ratio - $150B market cap  
3. **DHR** (Danaher): 1.254 ratio - $200B market cap
4. **NEE** (NextEra Energy): 1.226 ratio - $150B market cap
5. **INTC** (Intel): 1.226 ratio - $200B market cap
6. **TMO** (Thermo Fisher): 1.217 ratio - $200B market cap
7. **WMT** (Walmart): 1.188 ratio - $500B market cap
8. **ADP** (Automatic Data Processing): 1.171 ratio - $100B market cap
9. **UNP** (Union Pacific): 1.148 ratio - $150B market cap
10. **AMD** (Advanced Micro Devices): 1.137 ratio - $200B market cap

### Sector Insights
- **Healthcare** (UNH, TMO) shows strong upside bias
- **Technology** (TXN, INTC, AMD) shows mixed patterns
- **Energy** (NEE) shows upside bias
- **Consumer** (WMT) shows upside bias
- **Financials** (ADP, UNP) show upside bias

### Interpretation
- **Ratio > 1.0**: Higher beta during positive market days (more sensitive to upside)
- **Ratio < 1.0**: Higher beta during negative market days (more sensitive to downside)
- **Largest ratios**: Companies that perform better in bull markets
- **Smallest ratios**: Companies that perform better in bear markets

## Methodology

### Data Collection
- **Market Proxy**: SPY (S&P 500 ETF)
- **Stock Universe**: 45 major US stocks + 50 S&P 500 companies
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
- Absolute difference between positive and negative betas (Positive - Negative)
- Positive values: Higher sensitivity in up markets
- Negative values: Higher sensitivity in down markets
- Larger bars indicate more asymmetric behavior

### 2. T-Test Results Visualization
![T-Test Results](https://github.com/chungderson/nonlinear-beta/raw/main/docs/t_test_results.png)

**Top Left**: Positive vs Negative Beta Scatter
- Each point represents a stock
- Red dashed line shows equal betas
- Green line shows mean difference

**Top Right**: Distribution of Beta Differences
- Histogram showing the spread of differences (Positive - Negative)
- Red line at zero (no difference)
- Green line shows mean difference
- Positive values: Higher upside sensitivity
- Negative values: Higher downside sensitivity

**Bottom Left**: Box Plot Comparison
- Side-by-side comparison of positive vs negative betas
- Shows median, quartiles, and outliers

**Bottom Right**: Top 10 Stocks by Beta Difference
- Horizontal bar chart of largest differences (Positive - Negative)
- Blue bars: Higher positive beta (more sensitive in up markets)
- Red bars: Higher negative beta (more sensitive in down markets)

### 3. S&P 500 Beta Asymmetry Heatmap
![S&P 500 Beta Heatmap](https://github.com/chungderson/nonlinear-beta/raw/main/docs/sp500_beta_heatmap.png)

**Market Cap Weighted Analysis**:
- Color-coded heatmap showing positive-to-negative beta ratios
- Companies weighted by market capitalization
- Top 50 S&P 500 companies analyzed
- Red colors: Higher positive beta (upside bias)
- Blue colors: Higher negative beta (downside bias)
- Text annotations show company symbol, beta ratio, and market cap

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
4. **Sector Patterns**: Healthcare and technology show distinct asymmetric patterns
5. **Market Cap Effects**: Larger companies show different asymmetry patterns than smaller ones

## Development Tools

This project was developed using:
- **Cursor IDE**: Advanced AI-powered code editor for rapid development and debugging
- **Claude Sonnet 4**: AI assistant for code generation, analysis, and documentation
- **GitHub**: Version control and repository hosting
- **Alpaca Markets API**: Financial data provider

*Built with modern AI-assisted development tools to accelerate research and analysis.*

---

**Note**: This analysis challenges traditional financial theory and should be used alongside other risk management tools. Past performance does not guarantee future results.
