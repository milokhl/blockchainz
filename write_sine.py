import os, sys
import pandas as pd
import numpy as np

def write_sine_price():
  num_days = 1000
  dates = pd.date_range('1/1/2000', periods=num_days)
  xvals = np.linspace(0, 10*2*3.14159, num=num_days)

  close_series = pd.Series(100 * np.sin(xvals) + 10, index=dates)
  close = close_series.values
  high = close + 5
  low = close - 5
  vol = np.random.rand(num_days)
  cap = close * vol

  df = pd.DataFrame()
  df['Date'] = dates
  df['Open'] = close
  df['High'] = high
  df['Low'] = low
  df['Close'] = close
  df['Volume'] = vol
  df['Market Cap'] = cap
  
  df.set_index('Date')

  coin_path = '../datasets/popular_coins'
  filepath = os.path.join(coin_path, 'sinewave_price.csv')
  df.to_csv(filepath, index=False)

write_sine_price()