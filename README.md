# Algorithmic Trading with Hidden Markov Models

## Project Description
This repository contains a quantitative trading architecture designed to dynamically identify market regimes and manage capital allocation. 

The statistical engine utilizes Hidden Markov Models (HMM) with Gaussian Mixture Model (GMM) distributions to overcome the limitations of normal distributions when analyzing financial returns (fat tails and extreme events).

## Core Features
Compared to standard educational models, this project introduces several advanced implementations:
* **3-State Modeling:** Moving beyond simple binary logic (Bull/Bear) to identify directional market regimes (Bull/Bear) and high/low volatility stagnation phases (Choppy markets).
* **Multivariate Input:** The Expectation-Maximization (Baum-Welch) algorithm is trained on an observation vector that includes both logarithmic returns (Total Return) and rolling volatility.
* **Execution Rules and Hysteresis:** Implementation of smoothing on filtered probabilities and asymmetric thresholds (e.g., switching only at > 80% confidence) to drastically reduce chattering and transaction costs in the walk-forward backtest.

## Tech Stack
* **Language:** Python
* **Machine Learning & Statistics:** `hmmlearn`, `scikit-learn`
* **Data Processing:** `pandas`, `numpy`, `yfinance`

## Repository Structure
* `src/`: Contains source modules for data loading, HMM computation, trading logic, and the backtest engine.
* `notebooks/`: Jupyter Notebooks used for visual data exploration and market regime plotting.
* `data/`: Local directory for datasets (ignored by Git).
