import os
from pathlib import Path

os.environ.setdefault('YF_DISABLE_CURL_CFFI', '1')

import yfinance as yf
import pandas as pd
import numpy as np

yf.set_tz_cache_location(str(Path(__file__).resolve().parents[1] / '.cache' / 'yfinance'))

def download_and_process_data(start_date='2004-11-19', end_date='2026-06-19'):
    spy_data = yf.Ticker('SPY').history(start=start_date, end=end_date)
    gld_data = yf.Ticker('GLD').history(start=start_date, end=end_date)
    
    # Create combined DataFrame extracting closing prices
    data = pd.DataFrame({
        'SPY_CLOSE': spy_data['Close'],
        'GLD_CLOSE': gld_data['Close']
    })
    
    # Weekly resample (Friday)
    # Remove timezone (tz-naive) to avoid conflicts with other libraries
    data.index = data.index.tz_localize(None) 
    weekly_data = data.resample('W-FRI').last()
    
    # Logarithmic returns
    weekly_data['SPY_RETURN'] = np.log(weekly_data['SPY_CLOSE'] / weekly_data['SPY_CLOSE'].shift(1))
    weekly_data['GLD_RETURN'] = np.log(weekly_data['GLD_CLOSE'] / weekly_data['GLD_CLOSE'].shift(1))
    weekly_data['SPY_SIMPLE_RETURN'] = np.exp(weekly_data['SPY_RETURN']) - 1
    weekly_data['GLD_SIMPLE_RETURN'] = np.exp(weekly_data['GLD_RETURN']) - 1
    
    # 4-week rolling volatility
    weekly_data['VOLATILITY'] = weekly_data['SPY_RETURN'].rolling(window=4).std()
    
    # Remove initial NaNs
    combined = weekly_data.dropna()
    return combined
if __name__ == "__main__":
    df = download_and_process_data()
    print(df.tail())
