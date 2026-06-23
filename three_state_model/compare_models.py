"""
Compare the 2-state and 3-state HMM strategies using the SAME rolling window.

For this comparison we force both strategies to use the same training window
(208 weeks), then evaluate SPY and the static 50/50 benchmark on
the exact same out-of-sample dates.

"""
from src.data_loader import download_and_process_data
from src.backtest import run_backtest
from src.performance_metrics import calculate_cagr, calculate_sharpe_ratio, get_max_drawdown
from three_state_model.backtest_3states import TRAIN_WINDOW, run_backtest_3states

COMPARISON_WINDOW = TRAIN_WINDOW


def metrics_on_returns(returns):
    """Compute CAGR, Sharpe and Max Drawdown from a weekly return series."""
    cumulative = (1 + returns).cumprod()
    cagr = calculate_cagr(cumulative)
    sharpe = calculate_sharpe_ratio(returns)
    mdd = get_max_drawdown(cumulative)
    return cagr, sharpe, mdd


def print_row(name, returns):
    cagr, sharpe, mdd = metrics_on_returns(returns)
    print(f"{name:<18} {cagr:>10.2%} {sharpe:>10.2f} {mdd:>14.2%}")


def benchmark_returns(strategy_df):
    """Return same-period benchmark series aligned with the strategy window."""
    return {
        'S&P 500': strategy_df['SPY_SIMPLE_RETURN'],
        '50/50 Buy&Hold': 0.5 * strategy_df['SPY_SIMPLE_RETURN'] + 0.5 * strategy_df['GLD_SIMPLE_RETURN'],
    }


def main():
    market_data = download_and_process_data()

    # Run both strategies with the same training window.
    res_2 = run_backtest(market_data, train_window=COMPARISON_WINDOW)
    res_3 = run_backtest_3states(market_data, train_window=COMPARISON_WINDOW)

    # Keep only the dates both backtests have in common.
    common = res_2.index.intersection(res_3.index)
    if common.empty:
        raise ValueError("No overlapping dates found between the two backtests.")
    res_2 = res_2.loc[common]
    res_3 = res_3.loc[common]

    comparison_series = {
        'HMM 2 states': res_2['STRATEGY_RETURN'],
        'HMM 3 states': res_3['STRATEGY_RETURN'],
        **benchmark_returns(res_2),
    }

    print("\n" + "=" * 56)
    print(
        f"SAME-WINDOW COMPARISON ({COMPARISON_WINDOW} weeks, "
        f"{common[0].date()} -> {common[-1].date()})"
    )
    print("=" * 56)
    print(f"{'Strategy':<18} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>14}")
    print("-" * 56)
    for name, returns in comparison_series.items():
        print_row(name, returns)
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
