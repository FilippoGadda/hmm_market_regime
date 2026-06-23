from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from .backtest import run_backtest
    from .data_loader import download_and_process_data
    from .performance_metrics import add_performance_columns
except ImportError:
    from backtest import run_backtest
    from data_loader import download_and_process_data
    from performance_metrics import add_performance_columns


def plot_equity_lines(df, save_path='equity_curve.png'):
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative return lines
    ax.plot(df.index, df['CUMULATIVE_STRATEGY'], label='HMM Strategy', color='green', linewidth=2)
    ax.plot(df.index, df['CUMULATIVE_SPY'], label='S&P 500 Buy & Hold', color='blue', alpha=0.6)
    ax.plot(df.index, df['CUMULATIVE_50_50'], label='50/50 Buy & Hold', color='orange', alpha=0.6)
    
    ax.set_title('Performance Comparison: Wealth Index vs Benchmarks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1 (Base = 1.0)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}x'))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.text(
        0.5,
        0.01,
        'Note: the chart shows cumulative portfolio growth, not annualized return.',
        ha='center',
        fontsize=9
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(save_path)
    print(f"\nGrafico salvato in '{save_path.resolve()}'.")
    plt.show()


def main():
    market_data = download_and_process_data()
    results = run_backtest(market_data)
    plot_equity_lines(add_performance_columns(results))


if __name__ == "__main__":
    main()
