# Uses 10 different actions as amounts, and a 'sign' output which controls buy/sell

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
  def __init__(self, data, params):
    self.steps = data.shape[0]
    self.timestep = 0
    self.signal = pd.Series(np.zeros(self.steps)) # stores actions at each timestep
    self.data = data

    self.capital = params['starting_capital']
    self.coin = params['starting_coin']
    self.trade_unit = self.capital / 50 # USD

  def simulateAction(self, action, sign):
    close_price = self.data[self.timestep][0]

    # prevent divide by zero bugs
    if close_price > 0.01:
      if sign > 0:
      	usd_amt = min(action * 0.1 * close_price, self.capital)
      	coin_amt = usd_amt / close_price
      	self.capital -= usd_amt
      	self.coin += coin_amt
      	self.signal[self.timestep] = coin_amt

      elif sign < 0:
      	coin_amt = min(self.coin, action * 0.1)
      	usd_amt = coin_amt * close_price
      	self.capital += usd_amt
      	self.coin -= coin_amt
      	self.signal[self.timestep] = -1 * coin_amt

      else:
      	print "Probably an error: sign=", sign
    self.timestep += 1

  def getClose(self, step=0):
  	return self.data[self.timestep + step][0]

  def getReward(self):
    reward = 0
    close = self.data[self.timestep][0]
    prev_close = self.data[self.timestep-1][0]
    reward += (close - prev_close) * self.coin

    # simulate small tx fee
    if abs(self.signal[self.timestep-1]) > 0:
    	reward -= 0.02 * abs(self.signal[self.timestep-1])
    return reward

  def getState(self):
    state = self.data[self.timestep]
    full_state = np.concatenate((state, np.array([self.capital, self.coin])))
    return full_state.reshape(1, 9)

  def __repr__(self):
    close = self.data[self.timestep][0]
    assets = self.capital + self.coin * close
    return 'Timestep: %d Capital: %f Coin: %f Close: %f Assets: %f' % (self.timestep, self.capital, self.coin, close, assets)


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


def countNanValues(df):
	return df.isnull().sum()


def evaluate_performance(data, model):
  CurrentState = State(data, params)
  totalReward = 0

  for step in range(data.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
    Q_values, sign = model.predict(state)
    action_id = np.argmax(Q_values)
    CurrentState.simulateAction(action_id, sign)
    reward = CurrentState.getReward()
    totalReward += reward

  return totalReward, CurrentState


# Params
TRAINING_DATA, BTC_DATA_MIN, BTC_DATA_DAY = load_data()
num_actions = 10
num_features = 9
epochs = 20
gamma = 0.95
epsilon = 1.0 # decreases over time
learning_progress = []
verbose = False
print_state = False
params = {'starting_capital': 100000, 'starting_coin': 0}

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(num_features,))
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(inputs)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
actions = Dense(num_actions, activation='linear', name='actions_output')(x)
sign = Dense(1, activation='tanh', name='sign_output')(x)
model = Model(inputs=inputs, outputs=[actions, sign])
rms = RMSprop()
model.compile(optimizer=rms, loss='mse')

signal_list = []
for epoch in range(epochs):
  CurrentState = State(TRAINING_DATA, params)

  for step in range(TRAINING_DATA.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
    if verbose: print('State:', state)
    Q_values, sign = model.predict(state)

    if verbose: print('Q_values:', Q_values)

    if (random.random() < epsilon):
      action_id = np.random.randint(0, num_actions)
      sign = np.random.choice([-1, 1])
    else:
      action_id = np.argmax(Q_values)

    if verbose: print('Taking action:', action_id, sign)

    # go to the next state by performing action
    CurrentState.simulateAction(action_id, sign)
    reward = CurrentState.getReward()
    if verbose: print('Reward:', reward)

    y = np.copy(Q_values)
    if CurrentState.timestep == (CurrentState.steps-1): # terminal state
      y[0][action_id] = reward # need to use 0 index because 2d array

    else:
      # reward + discounted reward at next state
      next_state = CurrentState.getState()
      Q_values_next, sign_next = model.predict(next_state)
      Q_max = np.max(Q_values_next)
      y[0][action_id] = reward + (gamma * Q_max)

    if verbose: print('Q_values_update:', y)

    if (CurrentState.getClose(step=0) - CurrentState.getClose(step=-1) > 0):
    	correct_sign = np.ones(1)
    else:
    	correct_sign = -1 * np.ones(1)

    model.fit(state, {'actions_output': y, 'sign_output': correct_sign}, batch_size=1, epochs=1, verbose=0)

  totalReward, finalState = evaluate_performance(TRAINING_DATA, model)
  signal_list.append(CurrentState.signal)
  print('Epoch #%d Total Reward: %f Epsilon: %f' % (epoch, totalReward, epsilon))
  print finalState

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