import pandas as pd
from tqdm import tqdm

from .hmm_3states import fit_hmm_3states

# With 3 states we estimate more parameters than the 2-state model,
# so we keep a longer default training window (4 years).
TRAIN_WINDOW = 208
PROBABILITY_COLUMNS = ['Prob_Low', 'Prob_Mid', 'Prob_High']


def run_backtest_3states(df, train_window=TRAIN_WINDOW):
    """Run the walk-forward backtest using a 3-state HMM."""
    print(f"\nStarting 3-State Walk-Forward Backtest (Window: {train_window} weeks)...")

    probability_records = []

    for end_idx in tqdm(range(train_window, len(df)), desc="Running Walk-Forward (3 states)"):

        # Slice the training data (the past 'train_window' weeks)
        train_slice = df.iloc[end_idx - train_window : end_idx].copy()
        target_date = df.index[end_idx]

        # Fit the model on this slice and get probabilities
        fitted_slice = fit_hmm_3states(train_slice)

        # Extract the probabilities for the most recent week
        latest_probabilities = fitted_slice[PROBABILITY_COLUMNS].iloc[-1]

        probability_records.append({
            'Date': target_date,
            **latest_probabilities.to_dict(),
        })

    df_probs = pd.DataFrame(probability_records).set_index('Date')

    # Join probabilities with the original data
    backtest_df = df.join(df_probs, how='inner')

    # Define weights
    # The weights always sum to 1 (fully invested).
    backtest_df['WEIGHT_SPY'] = backtest_df['Prob_Low'] + 0.5 * backtest_df['Prob_Mid']
    backtest_df['WEIGHT_GLD'] = backtest_df['Prob_High'] + 0.5 * backtest_df['Prob_Mid']

    # Calculate strategy returns
    backtest_df['STRATEGY_RETURN'] = (
        backtest_df['WEIGHT_SPY'] * backtest_df['SPY_SIMPLE_RETURN'] +
        backtest_df['WEIGHT_GLD'] * backtest_df['GLD_SIMPLE_RETURN']
    )

    return backtest_df.dropna()
