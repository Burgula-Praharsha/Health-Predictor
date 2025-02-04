import os
import pickle  # pre-trained model loading
import streamlit as st  # web app
from streamlit_option_menu import option_menu
import datetime  # Import datetime for timestamp

st.set_page_config(page_title='Heart Disease Prediction',
                   layout='wide',
                   page_icon="❤️")

# Load the pre-trained model
heart_disease_model = pickle.load(open(r"C:\Users\burgu\Downloads\heart_disease_model.sav", 'rb'))

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Heart Disease Prediction'],
                           menu_icon='hospital-fill', icons=['heart'], default_index=0)

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)
    
    # Inputs
    with col1:
        age = st.text_input('Age of the person')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
    with col3:
        chestpain = st.text_input('Chest Pain Type (0-3)')
    with col1:
        restingbp = st.text_input('Resting Blood Pressure')
    with col2:
        cholesterol = st.text_input('Serum Cholesterol (mg/dL)')
    with col3:
        fastingbloodsugar = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False)')
    with col1:
        restingecg = st.text_input('Resting ECG Results (0-2)')
    with col2:
        maxheartrate = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exerciseangina = st.text_input('Exercise-Induced Angina (1 = Yes, 0 = No)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')
    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy')
    with col1:
        thalassemia = st.text_input('Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)')

# Initialize diagnosis variable
heart_diagnosis = ''

if st.button('Heart Disease Test Result'):
    try:
        # Collect inputs into a list and convert to floats
        user_input = [age, sex, chestpain, restingbp, cholesterol, fastingbloodsugar,
                      restingecg, maxheartrate, exerciseangina, oldpeak, slope, ca, thalassemia]
        user_input = [float(x) for x in user_input]
        
        # Make prediction
        heart_prediction = heart_disease_model.predict([user_input])
        
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has heart disease'
        else:
            heart_diagnosis = 'The person does not have heart disease'
            
            # Save the result to a file
            result_file = r"C:\Users\burgu\Downloads\heart_disease_prediction_results.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_file, 'a') as file:
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Input Data: {user_input}\n")
                file.write(f"Prediction Result: {heart_diagnosis}\n\n")
    except ValueError:
        heart_diagnosis = 'Please enter valid numerical values for all fields'

# Display the result
st.success(heart_diagnosis)
