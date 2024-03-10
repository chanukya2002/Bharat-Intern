# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the Boston House Prices dataset
boston = load_boston()

# Create a DataFrame from the dataset
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target

# Split the dataset into features and target
X = boston_df.drop('PRICE', axis=1)
y = boston_df['PRICE']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model using scikit-learn
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
lr_predictions = lr_model.predict(X_test)

# Evaluate the model
lr_mse = mean_squared_error(y_test, lr_predictions)
print("Linear Regression Mean Squared Error:", lr_mse)

# Neural Network model using TensorFlow
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
nn_loss, nn_mae = model.evaluate(X_test, y_test)
print("Neural Network Mean Squared Error:", nn_loss)

# Making predictions using the neural network model
nn_predictions = model.predict(X_test)

# Display a few predictions
for i in range(5):
    print("Predicted Price:", nn_predictions[i][0], "| True Price:", y_test.iloc[i])
