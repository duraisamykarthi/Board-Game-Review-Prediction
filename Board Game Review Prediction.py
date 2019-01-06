# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:23:45 2019

@author: KARTHI
"""
import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

# import necessary packages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the Data
games = pandas.read_csv("games.csv")

# print the names of the columns in games
print(games.columns)
print(games.shape)

# Make a histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])
plt.show()

# Print the first row of all the games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

# Print the first row of the games with scores greater than 0
print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews
games = games[games["users_rated"] > 0]

# Remove any rows with missing values
games = games.dropna(axis = 0)

# make a histogram of all the average ratings
plt.hist(games["average_rating"])
plt.show()

# Correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)

plt.show()

# Get all the columns from the dataframe
columns = games.columns.tolist()

# Filter the columns to remove data wedo not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating","type","name","id"]]
 
# store the variable we'll be predicting on
target = "average_rating"

# Generate training and testing datasets
from sklearn.model_selection import train_test_split

# Generate training set
train = games.sample(frac = 0.8, random_state = 1)

# Select anything not in the training set and put it in test
test = games.loc[~games.index.isin(train.index)]

# Print shapes
print(train.shape)
print(test.shape)

# import linear regrassion model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression()

# Fit the model the training data
LR.fit(train[columns], train[target])

# Generate predictions for the test set
predictions = LR.predict(test[columns])

# Compute the error between our test prediction and actual values
print(mean_squared_error(predictions, test[target]))

# Linear Regression is not perfectly fit.because perfect fit means mean_squared_error is equal to zero.

# import the Random forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

# Fit to the data
RFR.fit(train[columns], train[target])

# make predictions
predictions = RFR.predict(test[columns])

# Compute the error between our test our test predictions and actual values
print(mean_squared_error(predictions, test[target]))

test[columns].iloc[0]

# Make prediction with both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

# Print out the predictions
print(rating_LR)
print(rating_RFR)

print(test[target].iloc[0])