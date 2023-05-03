from ast import parse
from tkinter.tix import WINDOW
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import math
import sys
import argparse
import random


import warnings
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

#changes the data into a format that is compatible with LSTM
# something like this
#          X               y
# [[[1], [2], [3]]]       [4]
# [[[2], [3], [4]]]       [5]
# [[[3], [4], [5]]]       [6]

# depending on the window size
# the X list size will vary
def df_to_X_y(df, window_size=5):
  X = []
  y = []
  for i in range(len(df)-window_size):
    row = df[i:i+window_size]
    X.append(row)
    label = df[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)


def GetData(filename, labelname, timestep):
  df = pd.read_csv(filename)
  label = df[labelname]
  years = df[timestep]
  return label, years


def PreprocessData(data, window):
  WINDOW_SIZE = window
  scaler = MinMaxScaler()
  #scale the data
  data = np.array(data)
  data = data.reshape(-1,1)

  scaler.fit(data)
  scaled_temp = scaler.transform(data)

  return scaled_temp, scaler

def MakeModel(my_args, WINDOW_SIZE):
  # define model
  model1 = Sequential()
  model1.add(LSTM(50, activation=my_args.activation, recurrent_activation= my_args.recurrent_activation, \
    dropout=my_args.dropout, unit_forget_bias=my_args.unit_forget_gate_bias, return_sequences=True, input_shape=(WINDOW_SIZE, 1)))
  model1.add(Dropout(0.2))
  model1.add(LSTM(100, return_sequences=True))
  model1.add(Dropout(0.2))
  model1.add(LSTM(200, return_sequences=True))
  model1.add(Dropout(0.2))
  model1.add(LSTM(400, return_sequences=True))
  model1.add(Dropout(0.2))
  model1.add(LSTM(200, return_sequences=True))
  model1.add(Dropout(0.2))
  model1.add(LSTM(100))
  model1.add(Dropout(0.2))
  #model1.add(Dense(100, activation='relu'))
  #model1.add(Dense(100, activation='relu'))
  #model1.add(Dense(50, activation='relu'))
  model1.add(Dense(1))

  model1.summary()
  model1.compile(loss='mse', optimizer='adam')

  return model1

def randomizeData(X, y):
  data = []
  for i in range(len(X)):
    data.append([X[i], y[i]])
  np.random.shuffle(data)
  x1 = []
  y1 = []
  for i in range(len(data)):
    x1.append((data[i][0]))
    y1.append((data[i][1]))

  return np.array(x1), np.array(y1)


def GetScaledData(my_args, window):
  csvName = my_args.data_file_name

  label_name = my_args.label_name
  time_step = my_args.time_step_name
  data, years = GetData(csvName, label_name, time_step)
  #print(years)
  scaled_data, scaler = PreprocessData(data, window)

  X1, y1 = df_to_X_y(scaled_data, window)

  if my_args.shuffle and not my_args.action == "score":
    X1, y1 = randomizeData(X1, y1)
  return X1, y1, scaler, years

def UnscalePredictions(X_pred, scaler):
  y = X_pred.reshape(-1, 1)
  y = scaler.inverse_transform(y)
  y = y.flatten()
  return y

def SplitData(my_args, X1, y1):
  #print(len(X1))
  end_val = int((1 - my_args.testing_percentage) * len(X1))
  end_train = int((1 - my_args.validation_percentage) * end_val)
  #print(end_train, end_val)
  X_train1, y_train1 = X1[:end_train], y1[:end_train]
  X_val1, y_val1 = X1[end_train:end_val], y1[end_train:end_val]
  X_test1, y_test1 = X1[end_val:], y1[end_val:]

  return X_train1, y_train1, X_val1, y_val1, X_test1, y_test1

def DoFit(my_args, X_train1, y_train1, X_val1, y_val1, window):
  model1 = MakeModel(my_args, window)
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, callbacks=[early_stopping])

  model1.save(my_args.model_file)


def DoScore(my_args, X1, y, X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, scaler, years):
  model1 = load_model(my_args.model_file)


  all_predictions = model1.predict(X1).flatten()
  all_pred_unscaled = UnscalePredictions(all_predictions, scaler)
  y_all_unscaled = UnscalePredictions(y, scaler)

  train_predictions = model1.predict(X_train1).flatten()
  train_pred_unscaled = UnscalePredictions(train_predictions, scaler)
  y_train1_unscaled = UnscalePredictions(y_train1, scaler)

  val_predictions = model1.predict(X_val1).flatten()
  val_pred_unscaled = UnscalePredictions(val_predictions, scaler)
  y_val1_unscaled = UnscalePredictions(y_val1, scaler)

  test_predictions = model1.predict(X_test1).flatten()
  test_pred_unscaled = UnscalePredictions(test_predictions, scaler)
  y_test1_unscaled = UnscalePredictions(y_test1, scaler)


  trainScore = math.sqrt(mean_squared_error(y_train1_unscaled, train_pred_unscaled))
  print('Train Score: %.2f RMSE' % (trainScore))
  valScore = math.sqrt(mean_squared_error(y_val1_unscaled, val_pred_unscaled))
  print('Validation Score: %.2f RMSE' % (valScore))

  if my_args.show_testing_score and not my_args.action == "learn":
    testScore = math.sqrt(mean_squared_error(y_test1_unscaled, test_pred_unscaled))
    print('Test Score: %.2f RMSE' % (testScore))


    all_results = pd.DataFrame(data={'All Predictions':all_pred_unscaled, 'Actuals':y_all_unscaled})

    plt.plot(np.array(years[my_args.window:]), all_results['All Predictions'])
    plt.plot(np.array(years[my_args.window:]), all_results['Actuals'])
    title = input("Name for title? ")
    plt.title(title)
    #plt.plot(all_results['Actuals'])
    plt.legend(['Trained Predicted', 'Actual'])
    plt.show()

  train_results = pd.DataFrame(data={'Train Predictions':train_pred_unscaled, 'Actuals':y_train1_unscaled})
  plt.plot(train_results['Train Predictions'])
  plt.plot(train_results['Actuals'])



  val_results = pd.DataFrame(data={'Val Predictions':val_pred_unscaled, 'Actuals':y_val1_unscaled})
  plt.plot(val_results['Val Predictions'])
  plt.plot(val_results['Actuals'])
  
  if my_args.show_testing_score:
    test_results = pd.DataFrame(data={'Test Predictions':test_pred_unscaled, 'Actuals':y_test1_unscaled})
    plt.plot(test_results['Test Predictions'])
    plt.plot(test_results['Actuals'])
  

  
def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='LSTM')
    parser.add_argument('action', default='score',
                        choices=[ "learn", "score", "learnscore" ], 
                        nargs='?', help="desired action")
    
    parser.add_argument('--window',   '-w', default=1, type=int, help="window size")

    parser.add_argument('--testing-percentage', '-t', default=0.3,  type=float, help="percentage set aside for testing data?")
    parser.add_argument('--validation-percentage', '-v', default=0.3 ,  type=float, help="of the data not set aside for testing, what percentage will be validation data?")

    parser.add_argument('--show-testing-score', '-s', default=0,  type=int, help="show the testing score if 1")
    parser.add_argument('--shuffle', '-r', default=0, type = int)

    parser.add_argument('--data-file-name', '-d', default="", help="the actual data file")
    parser.add_argument('--model-file', '-m', default="", help="the file to save/load the model to")
    
    parser.add_argument('--label-name', '-l', default="", help="")
    parser.add_argument('--time-step-name', '-p', default="years", help="the column for time advancing eg. years, date etc")

    my_args = parser.parse_args(argv[1:])

    return my_args



def main(argv):
  random.seed(7)
  #get data into right demensions
  my_args = parse_args(argv)

  #["relu", "sigmoid", "tanh"]
  activation = ["sigmoid"]
  recurrent_activation = [ "tanh"]
  dropout = [0.0]
  ufgb = [True, False]

  my_args.activation = np.random.choice(activation)
  my_args.recurrent_activation = np.random.choice(recurrent_activation)
  my_args.dropout = np.random.choice(dropout)
  my_args.unit_forget_gate_bias = np.random.choice(ufgb)

  #print(my_args.activation, my_args.recurrent_activation, my_args.dropout, my_args.unit_forget_gate_bias)

  WINDOW_SIZE = my_args.window
  X1, y1, scaler, years = GetScaledData(my_args, WINDOW_SIZE)
  

  # determine training, validation, and testing sizes
  X_train1, y_train1, X_val1, y_val1, X_test1, y_test1 = SplitData(my_args, X1, y1)

  #FITING
  if my_args.action == "learn":
    DoFit(my_args, X_train1, y_train1, X_val1, y_val1, WINDOW_SIZE)
    DoScore(my_args, X1, y1, X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, scaler, years) #wont show test scores

  # SCORING
  if my_args.action == "score":
    DoScore(my_args, X1, y1, X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, scaler, years)

  
  #FIT_SCORE_RANDOM
  if my_args.action == "learnscore":
    DoFit(my_args, X_train1, y_train1, X_val1, y_val1, WINDOW_SIZE)
    DoScore(my_args, X1, y1, X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, scaler, years)


  #print(my_args.activation, my_args.recurrent_activation, my_args.dropout, my_args.unit_forget_gate_bias)


if __name__ == "__main__":
    main(sys.argv)
