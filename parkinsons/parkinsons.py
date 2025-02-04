import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Loading the Parkinson's dataset
parkinsons_data = pd.read_csv(r"C:\Users\burgu\OneDrive\Desktop\parkinsons.csv")

# Check and drop 'name' column
parkinsons_data = parkinsons_data.drop(columns=['name'])

# Separate features and target variable
X = parkinsons_data.drop(columns=['status'])
Y = parkinsons_data['status']

# Splitting the data into Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize RandomForest model with balanced class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, Y_train)

# Make Predictions
Y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of the training data:', training_data_accuracy)

# Accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of the test data:', test_data_accuracy)

# Apply cross-validation on the RandomForest model
cross_val_scores = cross_val_score(model, X_train, Y_train, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Average cross-validation score:", cross_val_scores.mean())

# Plot confusion matrix
sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)), annot=True, fmt='g', cmap='Blues', xticklabels=['Healthy', 'Parkinson\'s'], yticklabels=['Healthy', 'Parkinson\'s'])
plt.title('Confusion Matrix')
plt.show()

# Plot the ROC curve
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (1)
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % roc_auc_score(Y_test, y_prob))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guessing line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


import pickle

# Save the model  
filename = 'parkinsons_model.sav'  
pickle.dump(model, open(filename, 'wb'))

# Load the saved model  
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Example input data  
input_data = (119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 0.093, 0.190, 0.241, 0.260, 0.165, 0.320, 2.301, 0.284, 2.871, 0.362, 19.289, 0.416, 0.053, 0.020, 0.034)
# Convert input to numpy array  
input_data_as_numpy_array = np.asarray(input_data)  
# Reshape input data  
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  
# Make prediction  
prediction = loaded_model.predict(input_data_reshaped)  
print(prediction)  
# Print result  
if prediction[0] == 0:  
    print('Person is healthy')  
else:  
    print('Person has Parkinson\'s')
