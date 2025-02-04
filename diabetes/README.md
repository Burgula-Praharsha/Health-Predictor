# Diabetes Prediction using Machine Learning

## Overview
This project predicts whether a person has diabetes based on medical input data. 
It uses a **Support Vector Machine (SVM)** model and a **Streamlit** web interface for easy interaction.  
Using this web app, users can input different medical parameters to predict the likelihood of diabetes.

## Features
✅ Simple and interactive web app  
✅ Uses a trained **SVM model** for prediction  
✅ Predicts diabetes risk for different new input values  

## Dataset  
The dataset contains features like **Glucose, BMI, Age, Blood Pressure**, etc.    

## Model Used  
**Support Vector Machine (SVM)** for classification  
**Model Accuracy:**  
- **Training Data Accuracy:** 78.33%  
- **Test Data Accuracy:** 77.27%    

## Usage  
1️⃣ Open the web app.  
2️⃣ Enter medical parameters.  
3️⃣ Click "Predict" to check diabetes risk.  

## Installation  
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

## Run the application  
```bash
streamlit run diabetes_prediction_app.py
```