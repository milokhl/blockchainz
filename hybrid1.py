import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras import regularizers

from keras.layers import Input, Dense
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler

import random, os, sys

# from talib.abstract import *
import talib

from collections import deque

matplotlib.style.use('ggplot')


def plot_trades(price_series, signal_series, title='Trading Signal'):
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

  plt.title(title)
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

  def getBatch(self, batch_size, weighted=True):
    normalized_weights = self.weights / np.sum(self.weights)
    if weighted:
      indices = np.random.choice(len(self.weights), batch_size, p=normalized_weights)
    else:
      indices = np.random.choice(len(self.weights), batch_size)
    X = np.zeros((batch_size, self.num_features))
    y = np.zeros((batch_size, self.num_actions))
    ctr = 0
    for i in indices:
      # print 'X:', X.shape, 'Buffer item:', self.buffer[i][0].shape
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

    self.trade_unit = self.capital / 100 # USD

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
        usd_amt = min(self.trade_unit, self.capital)
        coin_amt = usd_amt / close_price
        self.capital -= usd_amt
        self.coin += coin_amt
        self.signal[self.timestep] = coin_amt

      # sell BTC
      else:
        usd_amt = min(self.coin * close_price, self.trade_unit)
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
      reward -= 0.002
    # else:
    #   # diversification penalty
    #   ratio = max((self.coin * close) / self.capital, self.capital / (self.coin * close))
    #   penalty = (min(ratio, 10.0) - 1) / 900.0 # if ratio 
    #   reward -= penalty

    return reward * 100

  def getState(self):
    state = self.data_norm[self.timestep]
    # print state
    # full_state = np.concatenate((state, np.array([float(self.capital) / self.params['starting_capital']])))
    # return full_state.reshape(1, 1, 11)
    return state.reshape(1, 1, 10)

  def __repr__(self):
    close = self.data[self.timestep][0]
    assets = self.capital + self.coin * close
    return 'Timestep: %d Capital: %f Coin: %f Close: %f Pvalue: %f' % (self.timestep, self.capital, self.coin, close, self.pvalue[self.timestep])


def load_data():
  coin_path = '../datasets/popular_coins'

  btc_file = 'bitcoin_price.csv'
  ltc_file = 'litecoin_price.csv'
  xrp_file = 'ripple_price.csv'
  iota_file = 'iota_price.csv'
  eth_file = 'ethereum_price.csv'

  data_dict = {}
  for file in [btc_file, ltc_file, xrp_file, iota_file, eth_file]:
    full_path = os.path.join(coin_path, file)
    df = pd.read_csv(full_path, sep=',', skiprows=0, header=0, index_col=0, parse_dates=True,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])

    df['Date'] = df.index
    df.index = pd.to_datetime(df.index, unit='d')
    df = df.sort_values(by='Date')
    df.fillna(method='ffill')
    close = df['Close'].values.astype('float')
    high = df['High'].values.astype('float')
    low = df['Low'].values.astype('float')

    vol = df['Volume'].values
    vol = [v.replace('-', '0') for v in vol]
    vol = np.array([v.replace(',', '') for v in vol]).astype('float')

    cap = df['Volume'].values
    cap = [c.replace('-', '0') for c in cap]
    cap = np.array([c.replace(',', '') for c in cap]).astype('float')

    sma15 = talib.SMA(close, timeperiod=15)
    sma60 = talib.SMA(close, timeperiod=60)
    rsi = talib.RSI(close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    data = np.column_stack((
      close,
      sma15,
      sma60,
      rsi,
      atr,
      df['Open'].values,
      high,
      low,
      vol,
      cap)
    )

    data = np.nan_to_num(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(data)

    data_dict[file] = (norm_data, data, df)

  return data_dict


def countNanValues(df):
	return df.isnull().sum()


def evaluate_performance(data, data_norm, model, df, verbose=False, plot=False):
  CurrentState = State(data, data_norm, params)
  totalReward = 0

  for step in range(data.shape[0]-1):
    state = CurrentState.getState()
    Q_values = model.predict(state)
    action_id = np.argmax(Q_values)
    if verbose: print '[Eval] State:', CurrentState
    if verbose: print '[Eval] Qvalues:', Q_values
    if verbose: print '[Eval] Action:', action_id
    CurrentState.simulateAction(action_id)
    reward = CurrentState.getReward()
    if verbose: print 'Reward:', reward
    totalReward += reward

  if plot: plot_trades(df['Close'], CurrentState.signal, title='Performance Evaluation (Deterministic)')
  return totalReward, CurrentState

def evaluate_performance_hybrid(data, data_norm, model, df, verbose=False, plot=False):
  CurrentState = State(data, data_norm, params)
  totalReward = 0

  for step in range(data.shape[0]-1):
    state = CurrentState.getState()
    state_prediction, state_embedding = model['prediction_model'].predict(state)
    Q_values = model['action_model'].predict(state_embedding.reshape(1, state_embedding.shape[2]))
    action_id = np.argmax(Q_values)
    if verbose: print '[Eval] State:', CurrentState
    if verbose: print '[Eval] Qvalues:', Q_values
    if verbose: print '[Eval] Action:', action_id
    CurrentState.simulateAction(action_id)
    reward = CurrentState.getReward()
    if verbose: print 'Reward:', reward
    totalReward += reward

  if plot: plot_trades(df['Close'], CurrentState.signal, title='Performance Evaluation (Deterministic)')
  return totalReward, CurrentState

# Params
ALL_DATA = load_data()
TRAINING_DATA_NORM, TRAINING_DATA, DATAFRAME = ALL_DATA['bitcoin_price.csv']
num_actions = 3
num_features = 11

iters = 10
epochs = 10
decay_epoch = 4
epsilon = 1.0

active_gamma = 0.95
passive_gamma = 0.4

learning_progress = []
verbose = False
print_state = False
params = {'starting_capital': 20000, 'starting_coin': 0.0}
hidden_units = 64
batch_size = 16
xp_window_size = 32

state_embedding_size=10

def build_model():
  inputs = Input(shape=(1, num_features,))
  x = LSTM(hidden_units, input_shape=(1, num_features), return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(inputs)
  x = LSTM(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
  x = LSTM(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
  x = LSTM(hidden_units, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
  x = Dense(hidden_units, activation='relu', kernel_initializer='lecun_uniform')(x)
  actions = Dense(num_actions, activation='linear', name='actions_output')(x)
  model = Model(inputs=inputs, outputs=actions)
  optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=optimizer, loss='mse')
  return model

def build_model_2():
  inputs = Input(shape=(1, num_features,))
  x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(inputs)
  x = Dense(32, activation='relu', kernel_initializer='lecun_uniform')(x)
  x = Dense(16, activation='relu', kernel_initializer='lecun_uniform')(x)
  x = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(x)
  x = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(x)
  x = LSTM(16, kernel_regularizer=regularizers.l2(0.01))(x)
  actions = Dense(num_actions, activation='linear', name='actions_output')(x)
  model = Model(inputs=inputs, outputs=actions)
  optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=optimizer, loss='mse')
  return model

def hybrid_model():
  num_features = 10
  state_embedding_size = 10

  inputs = Input(shape=(1, num_features,))
  x = LSTM(32, input_shape=(1, num_features), return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(inputs)
  x = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
  x = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
  state_embedding = Dense(state_embedding_size, activation='linear', name='state_emb')(x)
  x = Dense(16, activation='relu')(state_embedding)
  x = Dense(16, activation='relu')(x)
  state_prediction = Dense(num_features, activation='linear', name='state_pred')(x)

  # DQN
  dqn_input = Input(shape=(state_embedding_size,))
  dqn = Dense(16, activation='relu')(dqn_input)
  dqn = Dense(16, activation='relu')(dqn)
  dqn = Dense(16, activation='relu')(dqn)
  actions = Dense(num_actions, activation='linear')(dqn)

  prediction_model = Model(inputs=inputs, outputs=[state_prediction, state_embedding])
  action_model = Model(inputs=dqn_input, outputs=actions)

  optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  prediction_model.compile(optimizer=optimizer, loss='mse', loss_weights={'state_emb': 0, 'state_pred': 1.0})
  action_model.compile(optimizer=optimizer, loss='mse')
  return {'prediction_model': prediction_model, 'action_model': action_model}

# Model1 = build_model()
# Model2 = build_model()

# PredictModel = Model1
# UpdateModel = Model2
Model = hybrid_model()

# main loop
for iteration in range(iters): # each iteration switches between the two models
  print '------ ITERATION %d -------' % iteration
  epsilon = 1.0
  signal_list = []
  for epoch in range(epochs): # each epoch is one progression through the training data
    totalEpochReward = 0
    CurrentState = State(TRAINING_DATA, TRAINING_DATA_NORM, params)
    ExpReplay = ExperienceReplay(xp_window_size, state_embedding_size, num_actions)

    for step in range(TRAINING_DATA.shape[0]-1):
      state = CurrentState.getState()

      state_prediction, state_embedding = Model['prediction_model'].predict(state)
      Q_values = Model['action_model'].predict(state_embedding.reshape((1, state_embedding.shape[2])))

      if (random.random() < epsilon):
        action_id = np.random.randint(0, num_actions)
      else:
        action_id = np.argmax(Q_values)

      # go to the next state by performing action
      CurrentState.simulateAction(action_id)
      reward = CurrentState.getReward()
      totalEpochReward += reward

      y = np.copy(Q_values)

      # if terminal state, no discounted future reward
      if CurrentState.timestep == (CurrentState.steps-1):
        y[0][action_id] = reward # zero index needed because array is 2D

      # total reward = immediate reward + discounted reward at next state
      else:
        next_state = CurrentState.getState()

        # fit the prediction model
        Model['prediction_model'].fit(state, {'state_pred': next_state, 'state_emb': state_embedding}, batch_size=1, epochs=1, verbose=0)

        next_state_pred, next_state_emb = Model['prediction_model'].predict(next_state)
        Q_values_next = Model['action_model'].predict(next_state_emb.reshape((1, next_state_emb.shape[2])))
        Q_max = np.max(Q_values_next)

        if action_id == 0:
          y[0][action_id] = reward + (passive_gamma * Q_max)
        else:
          y[0][action_id] = reward + (active_gamma * Q_max)

      # get a batch from the experience replay and fit
      ExpReplay.bufferAppend((state_embedding, y, Q_values)) # state, actual, predicted

      if len(ExpReplay.buffer) >= batch_size:
        X_batch, y_batch = ExpReplay.getBatch(batch_size, weighted=False)
        Model['action_model'].fit(X_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0)

    signal_list.append(CurrentState.signal)
    print '[Main Loop] Epoch #%d Total Reward: %f Epsilon: %f' % (epoch, totalEpochReward, epsilon)

    if epoch >= (decay_epoch-1):
      if epsilon > 0.2:
        epsilon -= 1.0 / epochs
      # epsilon = 1.0 / (epoch - decay_epoch + 2)

  totalReward, finalState = evaluate_performance_hybrid(TRAINING_DATA, TRAINING_DATA_NORM, Model, DATAFRAME, plot=True)
  print '[Main Loop] Iteration #%d Total Reward: %f Epsilon: %f' % (iteration, totalReward, epsilon)
  print '[Main Loop] Final state:', finalState

# # main loop
# for iteration in range(iters): # each iteration switches between the two models
#   print '------ ITERATION %d -------' % iteration
#   epsilon = 1.0
#   signal_list = []
#   for epoch in range(epochs): # each epoch is one progression through the training data
#     totalEpochReward = 0
#     CurrentState = State(TRAINING_DATA, TRAINING_DATA_NORM, params)
#     ExpReplay = ExperienceReplay(xp_window_size, num_features, num_actions)

#     for step in range(TRAINING_DATA.shape[0]-1):
#       state = CurrentState.getState()
#       Q_values = PredictModel.predict(state)

#       if (random.random() < epsilon):
#         action_id = np.random.randint(0, num_actions)
#       else:
#         action_id = np.argmax(Q_values)

#       # go to the next state by performing action
#       CurrentState.simulateAction(action_id)
#       reward = CurrentState.getReward()
#       totalEpochReward += reward

#       y = np.copy(Q_values)

#       # if terminal state, no discounted future reward
#       if CurrentState.timestep == (CurrentState.steps-1):
#         y[0][action_id] = reward # zero index needed because array is 2D

#       # total reward = immediate reward + discounted reward at next state
#       else:
#         next_state = CurrentState.getState()
#         Q_values_next = PredictModel.predict(next_state)
#         Q_max = np.max(Q_values_next)

#         if action_id == 0:
#           y[0][action_id] = reward + (passive_gamma * Q_max)
#         else:
#           y[0][action_id] = reward + (active_gamma * Q_max)

#       # get a batch from the experience replay and fit
#       ExpReplay.bufferAppend((state, y, Q_values)) # state, actual, predicted

#       if len(ExpReplay.buffer) >= batch_size:
#         X_batch, y_batch = ExpReplay.getBatch(batch_size, weighted=False)
#         UpdateModel.fit(X_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0)

#     signal_list.append(CurrentState.signal)
#     print '[Main Loop] Epoch #%d Total Reward: %f Epsilon: %f' % (epoch, totalEpochReward, epsilon)

#     if epoch >= (decay_epoch-1):
#       if epsilon > 0.2:
#         epsilon -= 1.0 / epochs
#       # epsilon = 1.0 / (epoch - decay_epoch + 2)

#   totalReward, finalState = evaluate_performance(TRAINING_DATA, TRAINING_DATA_NORM, UpdateModel, DATAFRAME, plot=True)
#   print '[Main Loop] Iteration #%d Total Reward: %f Epsilon: %f' % (iteration, totalReward, epsilon)
#   print '[Main Loop] Final state:', finalState

#   # swap the pair of models
#   PredictModel, UpdateModel = UpdateModel, PredictModel
  # plot_trades(pd.Series(PRICE_SERIES['close']), signal_list[-1])

# plot_trades(pd.Series(PRICE_SERIES['close']), signal_list[-1])

# evaluate performance on litecoin
print "Testing with LTC"
LTC_TRAINING_DATA_NORM, LTC_TRAINING_DATA, LTC_DATAFRAME = ALL_DATA['litecoin_price.csv']
evaluate_performance(LTC_TRAINING_DATA, LTC_TRAINING_DATA_NORM, UpdateModel, LTC_DATAFRAME, plot=True)