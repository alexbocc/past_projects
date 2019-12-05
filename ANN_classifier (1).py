#!/usr/bin/env python
# coding: utf-8

# #### The following is an ANN classifier for determining the probability that a specific customer leaves a financial institution
# 
# #### Geographic and financial customer attributes are included altogether, as these are believed to have an impacting factor on the decison of leaving.

# In[12]:


#ANN implementation for classification problem


# Importing libraries

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing useful classes

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # For cat. data encoding
from sklearn.metrics import confusion_matrix


# Importing dataset

ds = pd.read_csv('Churn_Modelling.csv') # Importing dataset from parent dir
X = ds.iloc[:, 3:13].values # Selecting input array from dataset (independent variables: Customer indicators)
y = ds.iloc[:, 13].values # Selecting output vector from dataset (dependent variable: Categorical (binary) variable)


# Data pre-processing

labelencoder_X_1 = LabelEncoder() # Encoder object for encoding non-numerical data into numerical data (Non-numerical data values are encoded in columns 1 and 2)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # There are more than 2 countries in ds, so one-hot enc. needed in column 1
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # One variable removed after one-hot encoding because of data multi-collinearity


# Splitting dataset into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Test size is 0.2 (20%) of total ds


# Feature scaling for easing numerical computatations

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initialising ANN

# There are 11 nodes in input layer and 1 in output layer, hence (1+11)/2 nodes in hidden layers

def build_classifier():
    
    classifier = Sequential() # Classifier object for ANN structure
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) # Input layer has 11 neurons, rect. lin. unit function
    classifier.add(Dropout(p = 0.1)) # Avoid overfitting
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) # Second layer added
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer has 1 node, sigmoid funct. for 0-1 range values (probabilities)
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Using adam alg. for weight update
                                                                                                 # Binary value (YES OR NO) classification problem: binary_crossentropy loss func.
                                                                                                 # Accuracy metric for monitoring model performance                                                                                    
    return classifier


#Fitting training set to model

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100) #Weights are updated every 10 data samples
classifier.fit(X_train, y_train)

# Predicting the test set results

y_predicted = classifier.predict(X_test) # Extracting classifier predicitions
y_predicted = (y_predicted > 0.5) # Below 0.5 data value is transformed to 0, above 0.5 to 1


# ### After learning the correlations patterns in training dataset, the model predicts a set of customers likely to leave the institution.

# In[14]:


# Analysing ANN classifier accuracy

cm = confusion_matrix(y_test, y_predicted) # Summarises model prediction accuracy
cm


# ### According to the convolutional matrix, the model made 1711 correct predictions out of 2000

# In[ ]:


# Further analysis and improvements (Computationally heavy section)

accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1) # 10 fold cross validation
mean = accuracy.mean() 
variance = accuracy.std() # Mean and variance of accuracies



# Improving ANN classifier by tuning of parameters

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10) # Test ANN over grid of parameters (batch sizes, training alg.) defined above

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ # Best parameters placed in array
best_accuracy = grid_search.best_score_ # Best accuracy score obtained with parameters from best_parameters

