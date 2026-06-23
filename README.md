# HMM-Based Market Regime Switching Strategy

## Overview
In this project, I developed a quantitative trading framework that utilizes **Hidden Markov Models (HMM)** to detect latent market regimes. By distinguishing between "Bull/Quiet" and "Bear/Volatile" states, I designed a strategy that dynamically allocates capital between **SPY (S&P 500)** and **GLD (Gold)** to enhance risk-adjusted returns and minimize drawdowns during market turbulence.

As a research tool, I built this pipeline to emphasize methodological integrity, ensuring I avoid look-ahead bias through rigorous walk-forward validation.

## Performance Results
My backtesting results demonstrate that the HMM strategy significantly outperforms both the pure equity benchmark and a static 50/50 portfolio, particularly in terms of risk management (Max Drawdown).

| Metric | HMM Strategy | S&P 500 Buy & Hold | 50/50 Buy & Hold |
| :--- | :---: | :---: | :---: |
| **Annualized Return** | 13.99% | 8.99% | 9.37% |
| **Sharpe Ratio** | 0.92 | 0.56 | 0.75 |
| **Max Drawdown** | -25.32% | -58.36% | -30.94% |

*The strategy demonstrates a higher Sharpe Ratio and significantly lower Maximum Drawdown compared to the benchmarks, confirming the efficacy of the regime-switching logic during historical market stress periods.*

## Key Methodological Features

* **Regime Detection via HMM:** I employ Gaussian Hidden Markov Models to infer unobserved market states based on log returns and 4-week rolling volatility of the S&P 500.
* **Label Switching Correction:** I addressed the technical challenge of arbitrary HMM state labeling by implementing a systematic "Label Switching" logic. This ensures consistency by mapping states based on realized volatility metrics, guaranteeing that the "Risky" state is always correctly identified.
* **Walk-Forward Backtesting:** To ensure the model remains generalizable and realistic, I utilize a sliding window approach. I retrain the model continuously, which ensures that my decisions at time *t* are based only on information available prior to *t*.
* **Dynamic Asset Allocation:** The model outputs state probabilities, which I use as weights for portfolio rebalancing between SPY and GLD, acting as a tactical hedge against equity market downturns.

## Project Architecture

```text
hmm_market_regime/
├── src/
│   ├── data_loader.py       # Data fetching (yfinance), log-returns calculation, & rolling volatility
│   ├── hmm_model.py         # GaussianHMM implementation with custom Label Switching logic
│   ├── backtest.py          # Walk-forward engine ensuring no look-ahead bias
│   └── performance.py       # Metrics calculation (Sharpe Ratio, MDD) and equity curve plotting
├── requirements.txt         # Dependencies (pandas, hmmlearn, scikit-learn, etc.)
├── README.md                # This document
└── .gitignore               # Repository cleaning rules
## Technical Implementation

### Core Libraries I utilized:
* **`hmmlearn`**: For the Gaussian Hidden Markov Model implementation.
* **`pandas` & `numpy`**: For efficient data manipulation and vectorization.
* **`scikit-learn`**: For standardization of input features, which I found crucial for HMM convergence.
* **`yfinance`**: For reliable access to historical market data.
* **`matplotlib`**: For visualization of equity curves and strategy comparisons.

## Strategy Validation
I validate my strategy against two primary benchmarks:

* **S&P 500 Buy & Hold**: To measure alpha generation against the pure equity market.
* **50/50 Static Allocation**: To verify if my regime-switching logic adds value compared to a simple diversified static portfolio.
