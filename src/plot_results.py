import pandas as pd
import matplotlib.pyplot as plt
from backtest import run_backtest
from data_loader import download_and_process_data

def plot_equity_lines(df):
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative return lines
    plt.plot(df.index, df['CUMULATIVE_STRATEGY'], label='HMM Strategy', color='green', linewidth=2)
    plt.plot(df.index, df['CUMULATIVE_SPY'], label='S&P 500 Buy & Hold', color='blue', alpha=0.6)
    plt.plot(df.index, df['CUMULATIVE_50_50'], label='50/50 Buy & Hold', color='orange', alpha=0.6)
    
    plt.title('Performance Comparison: HMM Strategy vs Benchmarks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('equity_curve.png')
    print("\nGrafico salvato come 'equity_curve.png' nella cartella src.")
    plt.show()

if __name__ == "__main__":
    market_data = download_and_process_data()
    results = run_backtest(market_data)
    
    # Calculate cumulative columns required for the plot
    results['CUMULATIVE_STRATEGY'] = (1 + results['STRATEGY_RETURN']).cumprod()
    results['CUMULATIVE_SPY'] = (1 + results['SPY_RETURN']).cumprod()
    results['CUMULATIVE_50_50'] = (1 + (0.5 * results['SPY_RETURN'] + 0.5 * results['GLD_RETURN'])).cumprod()
    
    plot_equity_lines(results)