import os
import pickle  # pre-trained model loading
import streamlit as st  # web app
from streamlit_option_menu import option_menu
import datetime  # missing import

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

diabetes_model = pickle.load(open(r"C:\Users\burgu\Downloads\diabetes_model.sav", 'rb'))

with st.sidebar:
    selected = option_menu('Prediction of disease outbreak system',
                           ['Diabetes Prediction'],
                           menu_icon='hospital-fill', icons=['activity'], default_index=0)

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

diab_diagnosis = ''
if st.button('Diabetes Test Result'):
    try:
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        # Save the result to a file
        result_file = r"C:\Users\burgu\Downloads\diabetes_prediction_results.txt"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(result_file, 'a') as file:
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Input Data: {user_input}\n")
            file.write(f"Prediction Result: {diab_diagnosis}\n\n")
    except ValueError:
        diab_diagnosis = 'Please enter valid numerical values for all fields'

st.success(diab_diagnosis)
