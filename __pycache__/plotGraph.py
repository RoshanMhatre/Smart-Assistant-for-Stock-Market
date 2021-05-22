# This file was supposed to be used just for plotting but is not in use currently

import matplotlib.pyplot as plt
from train_model import train,valid,df,symbol

plt.style.use('fivethirtyeight')

def plotPredicted(title='Predictions',train=train,valid=valid):
    plt.figure(figsize=(12,6))
    plt.title('Predictions')
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Close Price (Rs.) ', fontsize=16)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'],loc='upper left')
    plt.show()

def plotGraph(title = symbol,dataFrame = df,column = 'Close'):
    plt.figure(figsize=(12,6))
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Rs. ',fontsize=10)
    plt.plot(df[column])
    plt.show()

print(plt.style.available)