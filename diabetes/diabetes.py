# Importing necessary libraries
import numpy as np  # Numerical Operations
import pandas as pd  # Data Manipulation
from sklearn.model_selection import train_test_split  # To train & test data
from sklearn import svm  # Importing SVM model
from sklearn.metrics import accuracy_score  # To check the Accuracy
import pickle  # For saving and loading the model

# === Data Collection & Analysis ===

# Loading the diabetes dataset from a CSV file
diabetes_dataset = pd.read_csv(r"C:\Users\burgu\OneDrive\Desktop\diabetes.csv")

# Display the first 5 rows of the dataset to get an overview of the data
print(diabetes_dataset.head())

# Checking for any missing values in the dataset
print(diabetes_dataset.isnull().sum())  # This will print the number of missing values in each column

# === Data Visualization ===

# Correlation heatmap to visualize the relationships between different features in the dataset
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.heatmap(diabetes_dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')  # Correlation matrix with annotations
plt.show()

# Distribution of the target variable 'Outcome' (Diabetes or not) using a count plot
sns.countplot(x='Outcome', data=diabetes_dataset)  # Shows count of people with or without diabetes
plt.show()

# === Data Preprocessing ===

# Target variable in dataset is Outcome
diabetes_dataset['Outcome']

# 0--->Non-diabetes & 1--->indicates diabetes
diabetes_dataset['Outcome'].value_counts()

#grouping the outcome
diabetes_dataset.groupby('Outcome').mean()

# Separate features (X) and target variable (Y)
X = diabetes_dataset.drop(columns=['Outcome'], axis=1)  # Features (excluding 'Outcome')
Y = diabetes_dataset['Outcome']  # Target variable ('Outcome')
print(X)
print(Y)

# Splitting the dataset into training and testing data (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#check
print(X.shape, X_train.shape, X_test.shape)

# === Model Training ===

# Create an instance of the Support Vector Machine (SVM) classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Train the classifier using the training data
classifier.fit(X_train, Y_train)

# === Model Evaluation ===

# Accuracy on the training data
X_train_prediction = classifier.predict(X_train)  # Predict on the training set
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)  # Compare predicted vs actual

# Accuracy on the test data
X_test_prediction = classifier.predict(X_test)  # Predict on the test set
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)  # Compare predicted vs actual

# Print the accuracy scores
print('Accuracy of the training data:', training_data_accuracy)
print('Accuracy of the test data:', test_data_accuracy)

# Print classification report (precision, recall, f1-score, support)
from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_prediction))

# Confusion matrix to visualize performance in terms of true positives, false positives, etc.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, X_test_prediction))

# === Model Saving ===

import pickle
# Saving the trained model using pickle
filename = 'diabetes_model.sav'  # The file where the model will be saved
pickle.dump(classifier, open(filename, 'wb'))  # Saving the model as a .sav file

# Loading the saved model (in case we need to load it later)
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# === Prediction on New Data ===

# Example input data for prediction (must have 8 features)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # Input should consist of 8 features
# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the data to match the model's input shape
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# Predicting the result using the loaded model
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
# Display the prediction result
if (prediction[0] == 0):
    print('Person not having diabetes')
else:
    print('Person having diabetes')

# === Display Feature Names ===

# Print the names of the features used in the model (to verify which columns are being used)
for column in X.columns:
    print(column)
