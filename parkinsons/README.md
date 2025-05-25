# Parkinson's Disease Prediction using Machine Learning

## Overview
This project predicts the likelihood of Parkinson's disease based on medical input data. 
It uses a **Random Forest** model and a **Streamlit** web interface for easy interaction.  
Using this web app, users can input different medical parameters to check their Parkinson's disease risk.

## Features
✅ Simple and interactive web app  
✅ Uses a trained **Random Forest** model for prediction  
✅ Predicts Parkinson's disease risk for different new input values  

## Dataset  
The dataset contains features like:  
- **MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)** (Fundamental frequencies)  
- **MDVP:Jitter(%)** (Jitter features)  
- **MDVP:Shimmer** (Shimmer features)  
- **NHR, HNR** (Noise-to-Harmonics and Harmonics-to-Noise Ratio)  
- **RPDE, spread1, spread2, D2** (Nonlinear dynamical features)  

## Model Used  
**Random Forest** for classification  
**Model Accuracy:**  
- **Training Data Accuracy:** 100%  
- **Test Data Accuracy:** 97.8%

### Confusion Matrix:
```
 [[29  1]
 [ 0 29]]
```

## Usage  
1️⃣ Open the web app.  
2️⃣ Enter medical parameters.  
3️⃣ Click "Predict" to check Parkinson's disease risk.  

## Installation  
1️⃣ Clone the repository.  
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

## Run the application  
```bash
streamlit run parkinsons_app.py
```
