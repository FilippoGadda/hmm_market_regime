import pandas as pd
import numpy as np
from tqdm import tqdm
try:
    from .data_loader import download_and_process_data
    from .hmm_model import fit_hmm_model
except ImportError:
    from data_loader import download_and_process_data
    from hmm_model import fit_hmm_model

def run_backtest(df, train_window=104):
    print(f"\nStarting Walk-Forward Backtest (Window: {train_window} weeks)...")
    
    # Storage for the results
    results = []
    
    # Iterate through the dataframe starting from the training window size
    for i in tqdm(range(train_window, len(df)), desc="Running Walk-Forward"):
        
        # 1. Slice the training data (the past 104 weeks)
        train_slice = df.iloc[i - train_window : i].copy()
        target_date = df.index[i]
        
        # 2. Fit the model on this slice and get probabilities
        res_slice = fit_hmm_model(train_slice)
        
        # 3. Extract the probabilities for the 'target week'
        # We take the last row of the result
        prob_risky = res_slice['Prob_Risky'].iloc[-1]
        prob_safe = res_slice['Prob_Safe'].iloc[-1]
        
        results.append({
            'Date': target_date,
            'Prob_Risky': prob_risky,
            'Prob_Safe': prob_safe
        })
    
    # Convert list to DataFrame
    df_probs = pd.DataFrame(results).set_index('Date')
    
    # 4. Join probabilities with the original data
    backtest_df = df.join(df_probs, how='inner')
    
    # 5. Define Allocation (Weights)
    # SPY weight = Prob_Safe, GLD weight = Prob_Risky
    backtest_df['WEIGHT_SPY'] = backtest_df['Prob_Safe']
    backtest_df['WEIGHT_GLD'] = backtest_df['Prob_Risky']
    
    # 6. Calculate Strategy Returns
    # Return = (Weight_SPY * SPY_Return) + (Weight_GLD * GLD_Return)
    backtest_df['STRATEGY_RETURN'] = (
        backtest_df['WEIGHT_SPY'] * backtest_df['SPY_SIMPLE_RETURN'] +
        backtest_df['WEIGHT_GLD'] * backtest_df['GLD_SIMPLE_RETURN']
    )
    
    return backtest_df.dropna()

if __name__ == "__main__":
    # Execution
    market_data = download_and_process_data()
    strategy_results = run_backtest(market_data, train_window=104)
    
    print("\nBacktest completed.")
    print(strategy_results[['WEIGHT_SPY', 'WEIGHT_GLD', 'STRATEGY_RETURN']].tail())
