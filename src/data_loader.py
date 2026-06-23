import os
from pathlib import Path

os.environ.setdefault('YF_DISABLE_CURL_CFFI', '1')

import numpy as np
import pandas as pd
import yfinance as yf

yf.set_tz_cache_location(str(Path(__file__).resolve().parents[1] / '.cache' / 'yfinance'))

DEFAULT_START_DATE = '2004-11-19'
DEFAULT_END_DATE = '2026-06-19'
WEEKLY_FREQUENCY = 'W-FRI'


def _download_history(ticker, start_date, end_date):
    return yf.Ticker(ticker).history(
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )


def download_and_process_data(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Download SPY and GLD data and build the weekly feature set."""
    spy_data = _download_history('SPY', start_date, end_date)
    gld_data = _download_history('GLD', start_date, end_date)
    
    # Create combined DataFrame extracting closing prices
    data = pd.DataFrame({
        'SPY_CLOSE': spy_data['Close'],
        'GLD_CLOSE': gld_data['Close']
    })
    
    # Weekly resample (Friday)
    # Remove timezone (tz-naive) to avoid conflicts with other libraries
    data.index = data.index.tz_localize(None) 
    weekly_data = data.resample(WEEKLY_FREQUENCY).last()
    
    # Logarithmic returns
    weekly_data['SPY_RETURN'] = np.log(weekly_data['SPY_CLOSE'] / weekly_data['SPY_CLOSE'].shift(1))
    weekly_data['GLD_RETURN'] = np.log(weekly_data['GLD_CLOSE'] / weekly_data['GLD_CLOSE'].shift(1))
    weekly_data['SPY_SIMPLE_RETURN'] = np.exp(weekly_data['SPY_RETURN']) - 1
    weekly_data['GLD_SIMPLE_RETURN'] = np.exp(weekly_data['GLD_RETURN']) - 1
    
    # 4-week rolling volatility
    weekly_data['VOLATILITY'] = weekly_data['SPY_RETURN'].rolling(window=4).std()
    
    # Remove initial NaNs
    return weekly_data.dropna()


def main():
    df = download_and_process_data()
    print(df.tail())


if __name__ == "__main__":
    main()
