# Task-3
import pandas as pd
import numpy as p

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn. linear model import LinearRegression from sklearn.metrics import mean_squared_error

# Load and Explore Data

data = pd.read_csv('kaggle datasets download -d bumba53
/advertisingcsv')
 print(data.head()) # Display the first few rows of the
 print(data.info()) # Get information about the dataset

# Data Preprocessing

data = data.dropna()
 data_encoded = pd.get_dummies (data)
 X = data_encoded. drop( 'sales', axis=1)
 y = data_encoded[ 'sales']
 X_train, X_test, y_train, y_test = train_test_split(X, y =0.2, random_state=42)
# Train the Model
 model = LinearRegression ()
 model. fit (X_train, y_train)

# Evaluate the Model
 y_pred = model.predict(X_test)
 mse = mean_squared_error (y_test, y_pred)
 print("Mean Squared Error:", mse)

# Visualize the results
 plt.scatter (y_test, y_pred)
