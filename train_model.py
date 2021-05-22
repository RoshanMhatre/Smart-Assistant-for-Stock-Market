# This file directly implements the video https://www.youtube.com/watch?v=QIUxPv5PJOY


import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math as m
import GetStock as gs
from datetime import date
from tabulate import tabulate
from assistant import speak, takeCommand, wishme
import sys

plt.style.use('fivethirtyeight')

wishme()
speak('Which stock are you interested in?')
company = takeCommand()

if company is None:
    speak('I suppose there was some noise while you were speaking. Would you give another try please?')
    raise Exception('You need to repeat')
symbol = gs.getSymbol(company.lower())
stat = gs.getFuture(symbol)

if stat == 1:
    feedback = f'{company} is a good choice for investing in for a few days to few weeks. \
I\'d recommend you have a look at the graph or the prediction chart to take your call.'
else:
    feedback = f'It looks like it is not really a good idea to invest in {company}, \
but looking at graphs and history you can take you call.'

speak(feedback)
print(feedback)

def ask_about_stocks():
    menu = 'You can either choose from looking at the graph, history or predictions.'
    close = 'If you want to end this analysis, say "close session."'
    speak(menu)
    speak(close)
    print(menu,close,sep='\n')
    query = takeCommand()
    print(query) 
    if 'show the graph' or 'show graph' in query:
        plotGraph()
    elif 'history' in query:
        print(tabulate(df,headers = 'keys', tablefmt='fancy_grid'))
    elif 'predictions' in query:
        plotPredicted()
        print(tabulate(valid,headers = 'keys', tablefmt='fancy_grid'))
    elif 'close session' in query:
        sys.exit()
    else:
        fdbk = 'Sorry, didn\'t catch you!'
        speak(fdbk)
        print(fdbk)

df = web.DataReader(symbol, data_source = 'yahoo',start ='2016-01-01',end= date.today())
data = df.filter(['Close']).values

def plotGraph(title = symbol,dataFrame = df,column = 'Close'):
    plt.figure(figsize=(12,6))
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Rs. ',fontsize=10)
    plt.plot(dataFrame[column])
    plt.show()    

# plotGraph()

data = df.filter(['Close'])

dataset = data.values

training_data_len = m.ceil(len(dataset)*0.8)

#scaling

scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

# training dataset
train_data = scaler_data[0:training_data_len,]
x_train, y_train = [],[]

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])


x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# Build a LSTM model
def buildModel(x_train):
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
  model.add(LSTM(50, return_sequences= False))
  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(optimizer='adam',loss = 'mean_squared_error')
  model.fit(x_train,y_train,batch_size=1, epochs=2)
  return model

# Test Dataset
test_data = scaler_data[training_data_len-60:,:]

x_test, y_test = [], dataset[training_data_len:,:]

for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

# Get Predicted values from model

predictions = buildModel(x_train).predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Plotting graph
def plotPredicted(title=f'Predictions/{symbol}',train=train,valid=valid):
    plt.figure(figsize=(12,6))
    plt.title('Predictions')
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Close Price (Rs.) ', fontsize=16)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'],loc='upper left')
    plt.show()

# plotPredicted()
while True:
    #Say Close Session for exiting the program
  ask_about_stocks()

