import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM

import random, os, sys

from talib.abstract import *

matplotlib.style.use('ggplot')

class State(object):
  def __init__(self, data):
    self.steps = data.shape[0]
    self.timestep = 0
    self.signal = pd.Series(np.zeros(self.steps)) # stores actions at each timestep
    self.data = data # each row contains (price, diff)

  def simulateAction(self, action):
    self.timestep += 1
    if (action == 0): # do nothing
      pass

    elif (action == 1):
      self.signal[self.timestep] = 10 # buy 10 shares

    elif (action == 2):
      self.signal[self.timestep] = -10 # sell 10 shares

  def getReward(self):
    if self.signal[self.timestep] == 0:
      return 0
    if self.signal[self.timestep] > 0 and self.data[self.timestep][1] > 0:
      return 10
    elif self.signal[self.timestep] < 0 and self.data[self.timestep][1] < 0:
      return 10
    else:
      return -10

  def getState(self):
    return self.data[self.timestep].reshape(1, 1, 7)

def load_data():
  btc_path = '../datasets/bitcoin'
  btc_file = 'coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv'
  full_path = os.path.join(btc_path, btc_file)

  df = pd.read_csv(full_path, sep=",", skiprows=0, header=0, index_col=0, parse_dates=True,
                       names=['timestamp', 'open', 'high', 'low', 'close', 'vol_btc', 'vol_usd', 'weighted_price'])
  df['timestamp'] = df.index
  df.index = pd.to_datetime(df.index, unit='s')
  daily_mean = df.resample('D').mean()

  close = daily_mean['close'].values
  diff = np.diff(close)
  diff = np.insert(diff, 0, 0)
  sma15 = SMA(daily_mean, timeperiod=15)
  sma60 = SMA(daily_mean, timeperiod=60)
  rsi = RSI(daily_mean, timeperiod=14)
  atr = ATR(daily_mean, timeperiod=14)

  data = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr))
  data = np.nan_to_num(data)
  return data, df, daily_mean

TRAINING_DATA, BTC_DATA_MIN, BTC_DATA_DAY = load_data()
# print BTC_DATA_MIN.isnull().sum()
# print BTC_DATA_DAY.isnull().sum()

# BTC_DATA_DAY.plot(y='close')
# plt.show()

def evaluate_performance(data, model):
  CurrentState = State(data)
  totalReward = 0

  for step in range(data.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
    Q_values = model.predict(state)
    action_id = np.argmax(Q_values)
    CurrentState.simulateAction(action_id)
    reward = CurrentState.getReward()
    totalReward += reward

  return totalReward

# Params
num_actions = 3
num_features = TRAINING_DATA.shape[1]
print 'Features:', num_features
epochs = 10
gamma = 0.95
epsilon = 1.0 # decreases over time
learning_progress = []
verbose = False

model = Sequential()
model.add(LSTM(16,
               input_shape=(1, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.5))

model.add(LSTM(16,
               input_shape=(1, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.5))

model.add(Dense(3, init='lecun_uniform'))
model.add(Activation('linear'))

adam = Adam()
model.compile(loss='mse', optimizer=adam)

signal_list = []
for epoch in range(epochs):
  CurrentState = State(TRAINING_DATA)
  if verbose: print('Epoch:', epoch)

  for step in range(TRAINING_DATA.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
    if verbose: print('State:', state)
    Q_values = model.predict(state)
    if verbose: print('Q_values:', Q_values)

    # choose random action with some probability
    if (random.random() < epsilon):
      action_id = np.random.randint(0, num_actions)
    else:
      action_id = np.argmax(Q_values)

    if verbose: print('Taking action:', action_id)

    # go to the next state by performing action
    CurrentState.simulateAction(action_id)
    reward = CurrentState.getReward()

    if verbose: print('Reward:', reward)

    y = np.copy(Q_values)
    if CurrentState.timestep == (CurrentState.steps-1): # terminal state
      y[0][action_id] = reward # need to use 0 index because 2d array

    else:
      # reward + discounted reward at next state
      next_state = CurrentState.getState()
      Q_values_next = model.predict(next_state)
      Q_max = np.max(Q_values_next)
      y[0][action_id] = reward + (gamma * Q_max)

    if verbose: print('Q_values_update:', y)

    model.fit(state, y, batch_size=1, epochs=1, verbose=0)

  totalReward = evaluate_performance(TRAINING_DATA, model)
  signal_list.append(CurrentState.signal)
  print('Epoch #%d Total Reward: %f Epsilon: %f' % (epoch, totalReward, epsilon))

  # slowly reduce epsilon as model gets smarter
  if epsilon > 0.1:
    epsilon -= (1.0/epochs)

def plot_trades(price_series, signal_series):
  price_series.plot(style='x-')

  none = signal_series == 0
  buy = signal_series > 0
  sell = signal_series < 0

  none_idx = none[none].index
  buy_idx = buy[buy].index
  sell_idx = sell[sell].index

  if none_idx.any():
    price_series[none_idx].plot(style='bo')

  if buy_idx.any():
    price_series[buy_idx].plot(style='ro')

  if sell_idx.any():
    price_series[sell_idx].plot(style='go')

  plt.show()

# plot the last (hopefully best) set of signals against price
plot_trades(pd.Series(BTC_DATA_DAY['close']), signal_list[-1])