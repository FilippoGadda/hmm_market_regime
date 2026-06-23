import pandas as pd
import numpy as np

WEEKS_PER_YEAR = 52


def get_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def calculate_cagr(cumulative_returns, periods_per_year=WEEKS_PER_YEAR):
    total_years = len(cumulative_returns) / periods_per_year
    return cumulative_returns.iloc[-1] ** (1 / total_years) - 1


def calculate_sharpe_ratio(returns, periods_per_year=WEEKS_PER_YEAR):
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def add_performance_columns(strategy_df):
    """Add cumulative-return columns for the strategy and benchmarks."""
    strategy_df = strategy_df.copy()
    strategy_df['RETURN_50_50'] = (
        (0.5 * strategy_df['SPY_SIMPLE_RETURN']) +
        (0.5 * strategy_df['GLD_SIMPLE_RETURN'])
    )
    strategy_df['CUMULATIVE_STRATEGY'] = (1 + strategy_df['STRATEGY_RETURN']).cumprod()
    strategy_df['CUMULATIVE_SPY'] = (1 + strategy_df['SPY_SIMPLE_RETURN']).cumprod()
    strategy_df['CUMULATIVE_50_50'] = (1 + strategy_df['RETURN_50_50']).cumprod()
    return strategy_df


def build_comparison_table(strategy_df):
    cagr_strategy = calculate_cagr(strategy_df['CUMULATIVE_STRATEGY'])
    sharpe_strategy = calculate_sharpe_ratio(strategy_df['STRATEGY_RETURN'])
    mdd_strategy = get_max_drawdown(strategy_df['CUMULATIVE_STRATEGY'])

    cagr_spy = calculate_cagr(strategy_df['CUMULATIVE_SPY'])
    sharpe_spy = calculate_sharpe_ratio(strategy_df['SPY_SIMPLE_RETURN'])
    mdd_spy = get_max_drawdown(strategy_df['CUMULATIVE_SPY'])

    cagr_50_50 = calculate_cagr(strategy_df['CUMULATIVE_50_50'])
    sharpe_50_50 = calculate_sharpe_ratio(strategy_df['RETURN_50_50'])
    mdd_50_50 = get_max_drawdown(strategy_df['CUMULATIVE_50_50'])

    comparison_data = {
        'Metric': ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy (HMM)': [f"{cagr_strategy:.2%}", f"{sharpe_strategy:.2f}", f"{mdd_strategy:.2%}"],
        'S&P 500': [f"{cagr_spy:.2%}", f"{sharpe_spy:.2f}", f"{mdd_spy:.2%}"],
        '50/50 Buy&Hold': [f"{cagr_50_50:.2%}", f"{sharpe_50_50:.2f}", f"{mdd_50_50:.2%}"]
    }
    return pd.DataFrame(comparison_data)


def calculate_metrics(strategy_df):
    """
    Calculates key performance metrics and compares HMM Strategy,
    S&P 500 Buy & Hold, and 50/50 Portfolio.
    """
    strategy_df = add_performance_columns(strategy_df)
    df_comparison = build_comparison_table(strategy_df)
    
    print("\n" + "="*65)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*65)
    print(df_comparison.to_string(index=False))
    print("="*65 + "\n")
    
    return strategy_df


def main():
    try:
        from .data_loader import download_and_process_data
        from .backtest import run_backtest
    except ImportError:
        from data_loader import download_and_process_data
        from backtest import run_backtest
    
    market_data = download_and_process_data()
    strategy_results = run_backtest(market_data)
    calculate_metrics(strategy_results)


if __name__ == "__main__":
    main()
