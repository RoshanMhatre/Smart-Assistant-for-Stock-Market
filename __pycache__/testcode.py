# This file has some changes from train_model file to oraganizethe code. So it becomes easy to make it in GUI.

import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math as m
import GetStock as gs
from tabulate import tabulate

plt.style.use('fivethirtyeight')

stock_name = 'TCS'
scaler = MinMaxScaler(feature_range=(0,1))

df = web.DataReader(gs.getSymbol(stock_name), data_source = 'yahoo',start = '2018-01-01',end= '2020-12-31')
dataset = df.filter(['Close']).values
training_data_len = m.ceil(len(dataset)*0.8)

def showHistory():
    print(tabulate(df[:-60],headers = 'keys'))

def showGraph():
    plt.figure(figsize=(12,6))
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Rs.',fontsize=10)
    plt.plot(df['Close'])
    plt.show()

#scaling
def getScaledData():
    scaler_data = scaler.fit_transform(dataset)
    return scaler_data

# training dataset
def getTrainingData():
    scaler_data = getScaledData()
    train_data = scaler_data[0:training_data_len,]
    x_train, y_train = [],[]

    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    x_train , y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    return x_train, y_train

def buildModel():
    # Build a LSTM model
    x_train, y_train = getTrainingData()

    model = Sequential()
    model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences= False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss = 'mean_squared_error')
    return model

def trainModel(model = buildModel()):
    x_train, y_train = getTrainingData()
    model.fit(x_train,y_train,batch_size=1, epochs=1)
    return model


''' ---------------- Testing Built Model-------------------- '''
def sampleData():
    # Test Dataset

    scaler_data = getScaledData()
    test_data = scaler_data[training_data_len-60:,:]

    x_test, y_test = [], dataset[training_data_len:,:]

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
    return x_test

def predictedValues(model):
    # Get Predicted values from model
    x_test = sampleData()
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Plotting data


# Plotting graph
def plotPredicted(train,valid):
    plt.figure(figsize=(12,6))
    plt.title('Predictions')
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Close Price USD($) ', fontsize=16)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'],loc='upper left')
    plt.show()

if __name__ == '__main__':
    
    #if user wants to see history
    # showHistory()

    #if user wants to see graph
    # showGraph()

    # Building a model here
    model = trainModel()
    
    # data = df.filter(['Close'])
    # train = data[:training_data_len]
    # valid = data[training_data_len:]

    # predictions = predictedValues(model)
    # valid['Predictions'] = predictions

    #if user asks for predicted graphs
    # plotPredicted(train,valid)

    #tabluated data
    # print(valid)

    tcsquote = web.DataReader(gs.getSymbol(stock_name), data_source = 'yahoo',start = '2018-01-01',end= '2020-12-31')

    new_df = tcsquote.filter(['Close'])
    last60 = new_df[-60:].values
    last60scaled = scaler.transform(last60)
    x_test = np.array(last60scaled)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    pred = model.predict(x_test)
    pred = scaler.inverse_transform(pred)

    print(pred)