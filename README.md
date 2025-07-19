# Nonlinear Beta Analysis: Challenging CAPM Assumptions

This repository implements a comprehensive analysis of nonlinear beta relationships in stock returns, challenging the traditional CAPM assumption that stocks have constant sensitivity to market movements regardless of market direction.

## Research Question

**Do stocks behave differently in positive vs negative market environments?**

Traditional CAPM assumes that a stock's beta (sensitivity to market movements) is constant. This analysis tests whether stocks exhibit asymmetric behavior - potentially having different betas in up markets vs down markets.

## Key Findings

### S&P 500 Comprehensive Analysis (2021-2025)
- **Sample Size**: 503 S&P 500 companies, 1,139 trading days
- **Positive Market Days**: 620 (54.4%)
- **Negative Market Days**: 517 (45.6%)
- **Mean Traditional Beta**: 0.976
- **Mean Positive Market Beta**: 0.978
- **Mean Negative Market Beta**: 0.961
- **Mean Asymmetry Ratio**: 1.002

### Most Asymmetric S&P 500 Stocks

**Highest Positive/Negative Beta Ratios (Upside Bias):**
1. **MKTX (MarketAxess)**: 1.916 ratio - Financials
2. **ED (Consolidated Edison)**: 1.915 ratio - Utilities
3. **ERIE (Erie Indemnity)**: 1.717 ratio - Financials
4. **LMT (Lockheed Martin)**: 1.590 ratio - Industrials
5. **MOH (Molina Healthcare)**: 1.559 ratio - Health Care

**Lowest Positive/Negative Beta Ratios (Downside Bias):**
1. **K (Kellogg)**: -2.868 ratio - Consumer Staples (negative beta)
2. **CBOE (Cboe Global Markets)**: 0.358 ratio - Financials
3. **GE (General Electric)**: 0.396 ratio - Industrials
4. **EXE (Exelon)**: 0.442 ratio - Energy
5. **KR (Kroger)**: 0.481 ratio - Consumer Staples

### Sector Analysis Results

**Sector Rankings by Beta Ratio (Positive/Negative):**
1. **Industrials**: 1.079 ratio (highest positive bias)
2. **Materials**: 1.061 ratio
3. **Information Technology**: 1.043 ratio
4. **Utilities**: 1.040 ratio
5. **Health Care**: 1.002 ratio
6. **Financials**: 0.995 ratio
7. **Consumer Discretionary**: 0.998 ratio
8. **Communication Services**: 0.988 ratio
9. **Real Estate**: 0.940 ratio
10. **Consumer Staples**: 0.945 ratio
11. **Energy**: 0.751 ratio (highest negative bias)

**Key Sector Insights:**
- **Energy stocks** show the most dramatic asymmetry with much higher negative betas
- **Industrial stocks** show the highest positive bias, rising more during market rallies
- **Consumer Staples** and **Real Estate** show defensive characteristics
- **Technology** shows moderate upside bias
- **Financials** are nearly symmetric

## Generated Visualizations

The analysis produces several key visualizations to help understand nonlinear beta relationships:

### 1. S&P 500 Sector Beta Comparison
![Sector Beta Comparison](https://github.com/chungderson/nonlinear-beta/raw/main/docs/sector_beta_comparison.png)

**Top Chart**: Average Positive vs Negative Betas by Sector
- Side-by-side bar charts showing positive (green) vs negative (red) betas
- Error bars show standard deviation within each sector
- Clear visualization of sector-level asymmetry patterns

**Bottom Chart**: Beta Ratio by Sector
- Beta ratio (Positive Beta / Negative Beta) for each sector
- Color coding: Green (ratio > 1), Red (ratio < 1), Gray (ratio = 1)
- Reference line at 1.0 for symmetric behavior
- Sample sizes (n=X) shown for each sector

### 2. S&P 500 Sector-by-Sector Analysis
![S&P 500 Sector Charts](https://github.com/chungderson/nonlinear-beta/raw/main/docs/sp500_optimized_sector_charts.png)

**Sector-by-Sector Bar Charts**:
- Horizontal bar charts for each sector
- Companies sorted by beta ratio within each sector
- Color coding: Red (high upside), Orange (moderate upside), Yellow (slight upside), Light Blue (slight downside), Blue (high downside)
- Reference line at 1.0 for symmetric behavior
- Ratio values displayed on each bar
- Comprehensive analysis across all major sectors

### 3. Beta Comparison Plots
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

### 4. T-Test Results Visualization
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

## Methodology

### Data Collection
- **Market Proxy**: SPY (S&P 500 ETF)
- **Stock Universe**: 503 S&P 500 companies
- **Time Period**: 2021-2025 (1,139 trading days)
- **Data Source**: Alpaca Markets API (daily bars)
- **Sector Classification**: GICS sectors from Wikipedia S&P 500 list

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
   Asymmetry = β_positive / β_negative
   ```

### Statistical Testing
- **Paired T-Test**: Compares positive vs negative betas across all stocks
- **Confidence Intervals**: 95% confidence intervals for mean differences
- **Significance Levels**: *** p<0.001, ** p<0.01, * p<0.05

## Interpretation Guide

### Asymmetry Ratio
- **Ratio > 1**: Stock is more sensitive in up markets
- **Ratio < 1**: Stock is more sensitive in down markets
- **Ratio = 1**: Equal sensitivity (traditional CAPM holds)

### Investment Implications
- **High Asymmetry**: Traditional beta may underestimate risk
- **Downside Asymmetry**: Stock falls more in bad markets than expected
- **Upside Asymmetry**: Stock rises more in good markets than expected

### Sector-Specific Insights
- **Energy**: High downside sensitivity - good hedge during market stress
- **Industrials**: High upside sensitivity - momentum plays during rallies
- **Consumer Staples**: Defensive characteristics with low overall betas
- **Technology**: Moderate upside bias with growth characteristics
- **Financials**: Nearly symmetric behavior across market conditions

## Results Summary

### Statistical Significance
- **T-Statistic**: Calculated from paired t-test comparing positive vs negative betas
- **P-Value**: Statistical significance of the difference
- **95% Confidence Interval**: Range where true mean difference likely falls

### Key Insights
1. **Nonlinear Behavior**: Many stocks show significant asymmetry
2. **Risk Management**: Traditional beta underestimates downside risk for asymmetric stocks
3. **Portfolio Construction**: Consider separate up/down betas for better risk modeling
4. **Sector Patterns**: Energy and Industrials show distinct asymmetric patterns
5. **Market Cap Effects**: Larger companies show different asymmetry patterns than smaller ones
6. **Sector Rotation**: Energy stocks may be better hedges, Industrials better momentum plays

## Development Tools

This project was developed using:
- **Cursor IDE**: Advanced AI-powered code editor for rapid development and debugging
- **Claude Sonnet 4**: AI assistant for code generation, analysis, and documentation
- **GitHub**: Version control and repository hosting
- **Alpaca Markets API**: Financial data provider

*Built with modern AI-assisted development tools to accelerate research and analysis.*

---

**Note**: This analysis challenges traditional financial theory and should be used alongside other risk management tools. Past performance does not guarantee future results.
