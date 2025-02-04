import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Loading the CSV data
heart_disease = pd.read_csv(r"C:\Users\burgu\OneDrive\Desktop\heart_disease.csv")

# Step 2: Checking the dataset
# Print first 5 rows
print(heart_disease.head())

# Print last 5 rows
print(heart_disease.tail())

# Number of rows and columns
print("Shape of dataset:", heart_disease.shape)

# Getting some info about data
print("Info about the dataset:")
print(heart_disease.info())

# Statistical measures about the data
print("Statistical summary:")
print(heart_disease.describe())

# Checking for missing values
print("Missing values in dataset:")
print(heart_disease.isnull().sum())

# Correlation matrix
print("Correlation matrix:")
print(heart_disease.corr())

# Step 3: Check the distribution of target variable
print("Target variable distribution:")
print(heart_disease['target'].value_counts())

# Step 4: Split data into features and target variable
X = heart_disease.drop(columns=['target'], axis=1)
Y = heart_disease['target']

# Step 5: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 6: Initialize Logistic Regression model
model = LogisticRegression()

# Step 7: Training the model with training data
model.fit(X_train, Y_train)

# Step 8: Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of the training data:', training_data_accuracy)

# Step 9: Accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of the test data:', test_data_accuracy)

# Step 10: Print classification report and confusion matrix
print("Classification report:")
print(classification_report(Y_test, X_test_prediction))

print("Confusion matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

import pickle
# Step 11: Save the trained model
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Step 12: Load the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Step 13: Making predictions with the model
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)  # should consist of 13 features in input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

# Result based on the prediction
if prediction[0] == 0:
    print('Person does not have heart disease')
else:
    print('Person has heart disease')

# Print column names
for column in X.columns:
    print(column)
