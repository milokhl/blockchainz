import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import random

matplotlib.style.use('ggplot')

class State(object):
  def __init__(self, data):
    self.steps = data.shape[0]
    self.timestep = 0
    self.signal = pd.Series(np.zeros(self.steps)) # stores actions at each timestep
    self.data = data # each row contains (price, diff)

  def simulateAction(self, action):
    self.timestep += 1
    if (action == 0):
      pass

    elif (action == 1):
      self.signal[self.timestep] = 10

    elif (action == 2):
      self.signal[self.timestep] = -10

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
    return self.data[self.timestep].reshape(1,2)

def load_data():
  num_days = 100
  dates = pd.date_range('1/1/2000', periods=num_days)
  xvals = np.linspace(0, 5*2*3.14159, num=num_days)
  prices = pd.Series(np.sin(xvals) + 1, index=dates)
  diffs = np.diff(prices)
  diffs = np.insert(diffs, 0, 0)
  return np.column_stack([prices, diffs]), prices

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
data, price_series = load_data()
epochs = 100
gamma = 0.9
epsilon = 1.0 # decreases over time
learning_progress = []
verbose = False

model = Sequential()
model.add(Dense(4, init='lecun_uniform', input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(num_actions, init='lecun_uniform'))
model.add(Activation('linear'))
rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

signal_list = []
for epoch in range(epochs):
  CurrentState = State(data)
  if verbose: print('Epoch:', epoch)

  for step in range(data.shape[0]-1):
    state = CurrentState.getState() # (price, diff)
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

  totalReward = evaluate_performance(data, model)
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
plot_trades(price_series, signal_list[-1])