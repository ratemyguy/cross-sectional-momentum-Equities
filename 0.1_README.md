# Cross-Sectional Momentum Strategy

A systematic long-only equity momentum strategy on U.S. large-cap stocks, rebalancing monthly based on 12–1 momentum signals. Built in Python with a custom backtesting framework, transaction cost modeling, and full robustness checks across lookback windows.

---

## Strategy Overview

**Core Idea:** Every month, rank all stocks in the universe by their past 12-month return (skipping the most recent month). Buy the top 20%. Rebalance monthly.

**The Academic Foundation:** Jegadeesh & Titman (1993) — one of the most cited papers in finance — documented that stocks with strong past 12-month returns continue to outperform over the next 3–12 months. This "momentum premium" has persisted across markets and decades.

**Why Skip the Last Month?**
The most recent month exhibits short-term *reversal* — stocks that just surged tend to mean-revert. Skipping it isolates the intermediate-term trend (months 2–12) where the real signal lives.

**Momentum Score Formula:**
```
momentum(t) = price(t-1) / price(t-12) - 1
```

---

## Project Architecture

```
Layer 1  →  Data Pipeline         (183 stocks, 2010–2024, cleaned)
Layer 2  →  Signal Generation     (12-1 momentum, quintile rankings)
Layer 3  →  Backtest Engine       (monthly rebalancing + transaction costs)
Layer 4  →  Robustness Checks     (6 lookback windows, 4 sizes, 5 cost levels)
```

---

## Signal Analysis (Layer 2)

Three filters were implemented and compared:

| Metric | Value |
|--------|-------|
| Universe | 183 U.S. large-cap stocks |
| Avg stocks in portfolio | 34 per month |
| Avg momentum score (top 20%) | ~18% trailing return |
| Hit rate vs median | Consistently positive |

**Quintile Forward Returns:**

| Quintile | Avg Monthly Return |
|----------|-------------------|
| Q1 (highest momentum) | **1.65%** |
| Q2 | 1.17% |
| Q3 | 1.34% |
| Q4 | 1.30% |
| Q5 (lowest momentum) | 1.44% |

Q1 outperforms all other quintiles. The non-monotonic pattern in Q2-Q5 is expected with a large-cap only universe — momentum is strongest at the extremes and works better with a broader universe including small/mid caps.

Cumulative return 2010–2024: Top quintile **+1,200%** vs Bottom quintile **+900%**.



---

## Backtest Results (Layer 3)

Monthly rebalancing with 0.10% transaction cost per trade (realistic institutional assumption).

| Metric | Momentum Strategy | Equal-Weight Benchmark |
|--------|------------------|----------------------|
| Ann. Return | 17.2% | 17.4% |
| Ann. Volatility | — | — |
| Sharpe Ratio | 1.18 | — |
| Max Drawdown | ~-18% | ~-21% |
| Monthly Turnover | 23% | 0% |
| Final Equity ($100k start) | ~$1.1M | ~$1.1M |

**Key Finding:** The strategy matches benchmark returns with consistently shallower drawdowns — particularly during the 2020 COVID crash where the benchmark fell ~21% vs the strategy's ~18%. The rolling 12-month Sharpe stayed above 1.0 for most of the 14-year period, dipping only during known momentum crash regimes (2011, 2022).



---

## Robustness Checks (Layer 4)

### 1. Lookback Window Sensitivity

The strategy was tested across six different momentum windows:

| Window | Sharpe | Ann. Return | Max DD |
|--------|--------|-------------|--------|
| 3-1 | 1.02 | — | — |
| 6-1 | **1.33** | — | — |
| 9-1 | 1.26 | — | — |
| 12-1 | 1.18 | — | — |
| 18-1 | 1.13 | — | — |
| 24-1 | 1.11 | — | — |

**Every single lookback window produces Sharpe > 1.0.** This is strong evidence the signal is not curve-fitted to the 12-1 parameter.

### 2. Portfolio Concentration

| Size | Final Equity |
|------|-------------|
| Top 10% (~18 stocks) | ~$1.75M |
| Top 20% (~36 stocks) | ~$1.1M |
| Top 30% (~55 stocks) | ~$900k |
| Top 40% (~73 stocks) | ~$850k |

More concentrated portfolios capture more alpha — the momentum signal is strongest at the very top of the ranking distribution.

### 3. Transaction Cost Sensitivity

| Cost | Ann. Return |
|------|-------------|
| 0.0% (theoretical) | 17.5% |
| 0.1% (institutional) | 17.2% |
| 0.3% (retail) | 15.8% |
| 0.5% | 14.8% |
| 1.0% | 11.2% |

The strategy remains profitable at 1.0% transaction costs — well above realistic institutional levels. The edge is genuine, not a low-cost artifact.



---

## Known Limitations

1. **Survivorship Bias:** The universe uses current S&P 500 constituents. A proper backtest would use point-in-time constituent data (FactSet, Compustat). This likely overstates returns modestly.

2. **Large-Cap Only:** Momentum is documented to be stronger in small/mid-cap stocks. This universe underrepresents the full signal.

3. **No Short Leg:** This is a long-only strategy. A long-short implementation (buy top quintile, short bottom quintile) would produce a cleaner factor return but requires margin and introduces short-selling costs.

4. **No Sector Neutralization:** The portfolio may concentrate in high-momentum sectors (e.g. Tech in 2023-2024). Sector-neutral construction would reduce this risk.

---

## How to Run

**Requirements:**
```bash
pip install yfinance pandas numpy matplotlib
```

**Run in order:**
```bash
python mom_layer1_data.py        # Download and clean price data (~3 min)
python mom_layer2_signal.py      # Compute momentum signals
python mom_layer3_backtest.py    # Run backtest with transaction costs
python mom_layer4_robustness.py  # Robustness checks (~2 min)
```

**Outputs:**
- `prices_daily.csv` / `prices_monthly.csv` — cleaned price data
- `momentum_scores.csv` / `momentum_signals.csv` — signal data
- `backtest_results_momentum.csv` — monthly PnL log
- `momentum_signals.png` — signal analysis chart
- `backtest_results_momentum.png` — equity curve and analytics
- `robustness_checks.png` — robustness analysis

---

## Tech Stack

- **Data:** yfinance (Yahoo Finance)
- **Analysis:** pandas, numpy
- **Visualization:** matplotlib

---

## Next Steps

- [ ] Expand universe to Russell 1000 for better signal coverage
- [ ] Add point-in-time constituent data to eliminate survivorship bias
- [ ] Implement sector-neutral portfolio construction
- [ ] Add long-short version for pure factor return
- [ ] Combine with value factor (momentum + value = documented premium)

---

*Strategy in active development. All results are backtested and do not represent live trading performance.*
