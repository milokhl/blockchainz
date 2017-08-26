import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM

from sklearn.preprocessing import MinMaxScaler

import random, os, sys

from talib.abstract import *

from collections import deque

matplotlib.style.use('ggplot')

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

class ExperienceReplay(object):
  def __init__(self, window_size, num_features, num_actions):
    self.window_size = window_size
    self.buffer = deque(maxlen=window_size)
    self.weights = deque(maxlen=window_size)
    self.num_features = num_features
    self.num_actions = num_actions

  def bufferAppend(self, item):
    self.buffer.append(item)
    error = np.linalg.norm(item[2] - item[1])
    self.weights.append(error)

  def getBatch(self, batch_size):
    normalized_weights = self.weights / np.sum(self.weights)
    indices = np.random.choice(len(self.weights), batch_size, p=normalized_weights)
    X = np.zeros((batch_size, 1, num_features))
    y = np.zeros((batch_size, num_actions))
    ctr = 0
    for i in indices:
      X[ctr] = self.buffer[i][0]
      y[ctr] = self.buffer[i][1]
      ctr += 1
    return X, y


class State(object):
  def __init__(self, data, data_norm, params):
    self.params = params

    self.steps = data.shape[0]
    self.timestep = 0
    self.signal = pd.Series(np.zeros(self.steps)) # stores actions at each timestep
    self.data = data
    self.data_norm = data_norm

    self.capital = params['starting_capital']
    self.coin = params['starting_coin']
    self.init_portfolio_value = self.getPortfolioValue() # timestep = 0

    self.pvalue = pd.Series(np.zeros(self.steps)) # stores the portfolio value over time
    self.pvalue[0] = self.init_portfolio_value

    self.trade_unit = self.capital / 50 # USD

  def getPortfolioValue(self):
    return self.capital + self.coin * self.data[self.timestep][0]

  def simulateAction(self, action):
    close_price = self.data[self.timestep][0]

    # prevent divide by zero bugs
    if close_price > 0.01:
      if action == 0:
        pass

      # buy BTC
      elif action == 1:
        usd_amt = min(100, self.capital)
        # usd_amt = 200
        coin_amt = usd_amt / close_price
        self.capital -= usd_amt
        self.coin += coin_amt
        self.signal[self.timestep] = coin_amt

      # sell BTC
      else:
        usd_amt = min(self.coin * close_price, 100)
        # usd_amt = 200
        coin_amt = usd_amt / close_price
      	self.capital += usd_amt
      	self.coin -= coin_amt
      	self.signal[self.timestep] = -1 * coin_amt

    self.timestep += 1
    self.pvalue[self.timestep] = self.getPortfolioValue()

  def getClose(self, step=0):
  	return self.data[self.timestep + step][0]

  def getReward(self):
    # Percentage increase / decrease in portfolio value.
    close = self.data[self.timestep][0]
    prev_close = self.data[self.timestep-1][0]
    reward = (self.pvalue[self.timestep] - self.pvalue[self.timestep-1]) / self.pvalue[self.timestep-1]

    # simulate small tx fee (normalized)
    if self.signal[self.timestep-1] > 0:
      reward -= (0.01 * abs(self.signal[self.timestep-1]) * prev_close) / self.pvalue[self.timestep-1]
    elif self.signal[self.timestep-1] < 0:
      reward -= 2.5 / self.pvalue[self.timestep-1] # coinbase $2.50 charge
    if (self.coin == 0 or self.capital == 0):
      reward -= 0.01

    return reward * 100

  def getState(self):
    state = self.data_norm[self.timestep]
    full_state = np.concatenate((state, np.array([self.capital / self.params['starting_capital']])))
    return full_state.reshape(1, 1, 11)

  def __repr__(self):
    close = self.data[self.timestep][0]
    assets = self.capital + self.coin * close
    return 'Timestep: %d Capital: %f Coin: %f Close: %f Assets: %f' % (self.timestep, self.capital, self.coin, close, self.pvalue[self.timestep])


def load_data():
  btc_path = '../datasets/bitcoin'
  btc_file = 'coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv'
  full_path = os.path.join(btc_path, btc_file)

  df = pd.read_csv(full_path, sep=",", skiprows=0, header=0, index_col=0, parse_dates=True,
                       names=['timestamp', 'open', 'high', 'low', 'close', 'vol_btc', 'vol_usd', 'weighted_price'])
  df['timestamp'] = df.index
  df.index = pd.to_datetime(df.index, unit='s')
  daily_mean = df.resample('D').mean()
  daily_mean = daily_mean.fillna(method='ffill')

  close = daily_mean['close'].values
  diff = np.diff(close)
  diff = np.insert(diff, 0, 0)
  sma15 = SMA(daily_mean, timeperiod=15)
  sma60 = SMA(daily_mean, timeperiod=60)
  rsi = RSI(daily_mean, timeperiod=14)
  atr = ATR(daily_mean, timeperiod=14)

  # data = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr))
  data = np.column_stack((close,
                          sma15,
                          sma60,
                          rsi,
                          atr,
                          daily_mean['open'].values,
                          daily_mean['high'].values,
                          daily_mean['low'].values,
                          daily_mean['vol_btc'].values,
                          daily_mean['vol_usd'].values))
  
  data = np.nan_to_num(data)

  scaler = MinMaxScaler(feature_range=(0, 1))
  norm_data = scaler.fit_transform(data)
  return norm_data, data, df, daily_mean


def countNanValues(df):
	return df.isnull().sum()


def evaluate_performance(data, data_norm, model, verbose=True):
  if verbose: print '----- Evaluating performance -----'
  CurrentState = State(data, data_norm, params)
  totalReward = 0

  for step in range(data.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
    Q_values = model.predict(state)
    action_id = np.argmax(Q_values)
    if verbose: print 'State:', CurrentState
    if verbose: print 'Qvalues:', Q_values
    if verbose: print 'Action:', action_id
    CurrentState.simulateAction(action_id)
    reward = CurrentState.getReward()
    if verbose: print 'Reward:', reward
    totalReward += reward

  plot_trades(pd.Series(BTC_DATA_DAY['close']), CurrentState.signal)
  return totalReward, CurrentState

# Params
TRAINING_DATA_NORM, TRAINING_DATA, BTC_DATA_MIN, BTC_DATA_DAY = load_data()
num_actions = 3
num_features = 11
epochs = 5
gamma = 0.93
epsilon = 1.0
learning_progress = []
verbose = False
print_state = False
params = {'starting_capital': 10000, 'starting_coin': 0.5}
iters = 10
hidden_units = 32
batch_size = 16
xp_window_size = 32

from keras.layers import Input, Dense
from keras.models import Model

def build_model():
  inputs = Input(shape=(1, num_features,))
  x = LSTM(hidden_units, input_shape=(1, num_features), return_sequences=True)(inputs)
  x = LSTM(hidden_units, return_sequences=True)(x)
  x = LSTM(hidden_units)(x)
  x = Dense(hidden_units, activation='relu', kernel_initializer='lecun_uniform')(x)
  actions = Dense(num_actions, activation='linear', name='actions_output')(x)
  model = Model(inputs=inputs, outputs=actions)
  optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=optimizer, loss='mse')
  return model

Model1 = build_model()
Model2 = build_model()

signal_list = []
PredictModel = Model1
UpdateModel = Model2

# main loop
for iteration in range(iters): # each iteration switches between the two models
  epsilon = 1.0
  for epoch in range(epochs): # each epoch is one progression through the training data
    CurrentState = State(TRAINING_DATA, TRAINING_DATA_NORM, params)
    ExpReplay = ExperienceReplay(xp_window_size, num_features, num_actions)

    for step in range(TRAINING_DATA.shape[0]-1):
      state = CurrentState.getState()
      Q_values = PredictModel.predict(state)

      if (random.random() < epsilon):
        action_id = np.random.randint(0, num_actions)
      else:
        action_id = np.argmax(Q_values)

      # go to the next state by performing action
      CurrentState.simulateAction(action_id)
      reward = CurrentState.getReward()

      y = np.copy(Q_values)

      # if terminal state, no discounted future reward
      if CurrentState.timestep == (CurrentState.steps-1):
        y[0][action_id] = reward # zero index needed because array is 2D

      # total reward = immediate reward + discounted reward at next state
      else:
        next_state = CurrentState.getState()
        Q_values_next = PredictModel.predict(next_state)
        Q_max = np.max(Q_values_next)
        y[0][action_id] = reward + (gamma * Q_max)

      # get a batch from the experience replay and fit
      ExpReplay.bufferAppend((state, y, Q_values)) # state, actual, predicted

      if len(ExpReplay.buffer) >= batch_size:
        X_batch, y_batch = ExpReplay.getBatch(batch_size)
        UpdateModel.fit(X_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0)

    totalReward, finalState = evaluate_performance(TRAINING_DATA, TRAINING_DATA_NORM, UpdateModel)
    signal_list.append(CurrentState.signal)
    print('Epoch #%d Total Reward: %f Epsilon: %f' % (epoch, totalReward, epsilon))
    print finalState

    # slowly reduce epsilon as model gets smarter
    if epsilon > 0.1:
      epsilon -= (1.0/epochs)

  # swap the pair of models
  PredictModel, UpdateModel = UpdateModel, PredictModel

# plot the last (hopefully best) set of signals against price
plot_trades(pd.Series(BTC_DATA_DAY['close']), signal_list[-1])