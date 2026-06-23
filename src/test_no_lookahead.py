"""
Test 1 - No look-ahead bias.

Idea: the decision for a given week must depend only on past data. If truncating the future data does NOT change the weights computed in the past, 
then no future information is "leaking" into the decisions.

"""
import pandas as pd

from src.data_loader import download_and_process_data
from src.backtest import run_backtest


def test_no_lookahead():
    df = download_and_process_data()

    # Backtest on the full dataset
    res_full = run_backtest(df.copy(), train_window=104)

    # Backtest on the first half of the data (the "future" is cut off)
    cutoff = len(df) // 2
    res_trunc = run_backtest(df.iloc[:cutoff].copy(), train_window=104)

    # On the shared dates the weights MUST be identical.
    # If they change, the future influences the past -> look-ahead bias.
    common = res_trunc.index.intersection(res_full.index)
    pd.testing.assert_series_equal(
        res_full.loc[common, "WEIGHT_SPY"],
        res_trunc.loc[common, "WEIGHT_SPY"],
        check_names=False,
    )


if __name__ == "__main__":
    test_no_lookahead()
    print("No look-ahead bias detected.")
