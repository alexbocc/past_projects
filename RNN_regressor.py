#!/usr/bin/env python
# coding: utf-8

# In[1]:


#RNN implementation


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing essential classes

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# Importing training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


# Feature Scaling

sc = MinMaxScaler(feature_range = (0, 1)) # Data set values are normalised to numbers between 0 and 1 (correlations preserved)
training_set_scaled = sc.fit_transform(training_set)


# Data structure with 60 timesteps and 1 output (Prediction is based on 60 previous values)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) 
    y_train.append(training_set_scaled[i, 0]) # y is determined by 60 values previous to i (STM)
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# RNN structure


regressor = Sequential() # Regression problem
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # Adding first LSTM layer with dropout regularisation
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True)) # Adding a second LSTM layer
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True)) # ...
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1)) # Output layer (Contains predicted numerical value by regressor)


# Compiling and fitting RNN regressor
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Good error function for regression problem is m.s.e.
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Real prices for January 2017 

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Predicting prices for January 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # Total ds needed for predicting
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values 
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Next prediction is always based on 60 previous observations and test dataset input
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) 
X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Normalised value is transformed back


# In[4]:


# Visualising the predictions of the RNN regressor

plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction for January financial days')
plt.xlabel('Time [Days]')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# ### The RNN regressor predicts the overall trends well; however, is not useful when the changes in price are highly non-linear. Over a 1 month period, acrually good predictions can be made on how the stock moved.

# In[ ]:




