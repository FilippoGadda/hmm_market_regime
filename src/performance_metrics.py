import pandas as pd
import numpy as np

def calculate_metrics(strategy_df):
    """
    Calculates key performance metrics and compares HMM Strategy,
    S&P 500 Buy & Hold, and 50/50 Portfolio.
    """
    # 1. Create columns for HMM Strategy
    strategy_df['CUMULATIVE_STRATEGY'] = (1 + strategy_df['STRATEGY_RETURN']).cumprod()
    
    # 2. Create columns for S&P 500 Benchmark
    strategy_df['CUMULATIVE_SPY'] = (1 + strategy_df['SPY_RETURN']).cumprod()
    
    # 3. Create columns for 50/50 Buy & Hold Benchmark
    strategy_df['RETURN_50_50'] = (0.5 * strategy_df['SPY_RETURN']) + (0.5 * strategy_df['GLD_RETURN'])
    strategy_df['CUMULATIVE_50_50'] = (1 + strategy_df['RETURN_50_50']).cumprod()
    
    # Helper for Annualized metrics
    total_years = len(strategy_df) / 52
    
    # Helper for Max Drawdown
    def get_max_drawdown(cum_returns):
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
    
    # --- Metrics for HMM Strategy ---
    cagr_strategy = (strategy_df['CUMULATIVE_STRATEGY'].iloc[-1])**(1/total_years) - 1
    sharpe_strategy = (strategy_df['STRATEGY_RETURN'].mean() / strategy_df['STRATEGY_RETURN'].std()) * np.sqrt(52)
    mdd_strategy = get_max_drawdown(strategy_df['CUMULATIVE_STRATEGY'])
    
    # --- Metrics for S&P 500 ---
    cagr_spy = (strategy_df['CUMULATIVE_SPY'].iloc[-1])**(1/total_years) - 1
    sharpe_spy = (strategy_df['SPY_RETURN'].mean() / strategy_df['SPY_RETURN'].std()) * np.sqrt(52)
    mdd_spy = get_max_drawdown(strategy_df['CUMULATIVE_SPY'])
    
    # --- Metrics for 50/50 ---
    cagr_50_50 = (strategy_df['CUMULATIVE_50_50'].iloc[-1])**(1/total_years) - 1
    sharpe_50_50 = (strategy_df['RETURN_50_50'].mean() / strategy_df['RETURN_50_50'].std()) * np.sqrt(52)
    mdd_50_50 = get_max_drawdown(strategy_df['CUMULATIVE_50_50'])
    
    # 4. Comparison Table
    comparison_data = {
        'Metric': ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy (HMM)': [f"{cagr_strategy:.2%}", f"{sharpe_strategy:.2f}", f"{mdd_strategy:.2%}"],
        'S&P 500': [f"{cagr_spy:.2%}", f"{sharpe_spy:.2f}", f"{mdd_spy:.2%}"],
        '50/50 Buy&Hold': [f"{cagr_50_50:.2%}", f"{sharpe_50_50:.2f}", f"{mdd_50_50:.2%}"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*65)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*65)
    print(df_comparison.to_string(index=False))
    print("="*65 + "\n")
    
    return strategy_df

if __name__ == "__main__":
    from data_loader import download_and_process_data
    from backtest import run_backtest
    
    # Run the full pipeline
    market_data = download_and_process_data()
    strategy_results = run_backtest(market_data)
    
    # Calculate and print
    final_results = calculate_metrics(strategy_results)