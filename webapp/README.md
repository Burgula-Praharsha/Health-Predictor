## Disease Prediction using Machine Learning

### Overview
This repository contains three machine learning models to predict the likelihood of diseases:

- **Diabetes Prediction (SVM)**
- **Heart Disease Prediction (Logistic Regression)**
- **Parkinson's Disease Prediction (Random Forest)**

All models are integrated with a **Streamlit** web app for easy interaction.

---

## Models

### 1. Diabetes Prediction
- **Model:** Support Vector Machine (SVM)
- **Accuracy:** 78.33% (Training), 77.27% (Test)
- **Features:** Glucose, BMI, Age, Blood Pressure and etc.

### 2. Heart Disease Prediction
- **Model:** Logistic Regression
- **Accuracy:** 85.12% (Training), 81.97% (Test)
- **Features:** Age, Cholesterol, Blood Pressure, Heart Rate and etc.

### 3. Parkinson's Disease Prediction
- **Model:** Random Forest
- **Accuracy:** 100% (Training), 97.8% (Test)
- **Features:** MDVP, Jitter, Shimmer, Harmonics-to-Noise Ratio and etc.

---

## Installation & Usage

### Clone the repository:
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction

##Run the app
streamlit run webapp.py
