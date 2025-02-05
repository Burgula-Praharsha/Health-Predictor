# Disease Prediction using Machine Learning

This repository contains machine learning models for predicting three diseases based on medical data. The goal is to provide automated predictions to help healthcare professionals in early disease diagnosis.

## Models Used:

1. **Diabetes Prediction (SVM)**
   - **Objective**: Predicts the likelihood of diabetes based on health parameters.
   - **Features**: Glucose levels, BMI, insulin levels, etc.
   - **Algorithm**: Implemented using Support Vector Machine (SVM).
   - **Accuracy**:
     - Training Data: 78.34%
     - Test Data: 77.27%

2. **Heart Disease Prediction (Logistic Regression)**
   - **Objective**: Predicts the risk of heart disease based on various health indicators.
   - **Features**: Age, cholesterol levels, blood pressure, ECG results, etc.
   - **Algorithm**: Implemented using Logistic Regression.
   - **Accuracy**:
     - Training Data: 85.12%
     - Test Data: 81.97%

3. **Parkinson’s Disease Prediction (Random Forest)**
   - **Objective**: Detects Parkinson’s disease based on voice measurements.
   - **Features**: Frequency variation, jitter, shimmer, etc.
   - **Algorithm**: Implemented using Random Forest Classifier.
   - **Accuracy**:
     - Training Data: 100%
     - Test Data: 94.87%

## Requirements
- Jupyter Notebook / VS code
- Libraries: `scikit-learn`, `streamlit`, `pandas`, `numpy`, `pickle`

## Usage
Upload patient data through the Streamlit web app.  
The model provides predictions based on trained machine learning algorithms.

