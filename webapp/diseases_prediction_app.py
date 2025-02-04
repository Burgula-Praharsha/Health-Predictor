import os
import pickle
import datetime
import streamlit as st
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(page_title='Disease Prediction',
                   layout='wide',
                   page_icon="🩺")

# Load the models (replace with the correct paths to your models)
heart_model = pickle.load(open(r"C:\Users\burgu\Downloads\heart_disease_model.sav", 'rb'))
diabetes_model = pickle.load(open(r"C:\Users\burgu\Downloads\diabetes_model.sav", 'rb'))
parkinsons_model = pickle.load(open(r"C:\Users\burgu\Downloads\parkinsons_model.sav", 'rb'))

# Set the title
st.title("Prediction of Disease Outbreaks")

# Option to select the model
selected = option_menu(
    "Select the Model", 
    ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
    icons=["activity", "heart", "mic"],
    menu_icon="cast", default_index=0
)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.subheader('Diabetes Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose')
    with col3:
        BloodPressure = st.text_input('Blood Pressure')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin')
    with col3:
        BMI = st.text_input('BMI')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2:
        Age = st.text_input('Age')

    diabetes_diagnosis = ''
    if st.button('Predict Diabetes'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        try:
            user_input = [float(x) for x in user_input]
            diabetes_prediction = diabetes_model.predict([user_input])

            if diabetes_prediction[0] == 1:
                diabetes_diagnosis = 'The person has diabetes'
            else:
                diabetes_diagnosis = 'The person does not have diabetes'

            # Save the result to a file
            result_file = r"C:\Users\burgu\Downloads\prediction_result.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_file, 'a') as file:
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Input Data: {user_input}\n")
                file.write(f"Prediction Result: {diabetes_diagnosis}\n\n")

        except ValueError:
            diabetes_diagnosis = 'Please enter valid numerical values for all fields'

    st.success(diabetes_diagnosis)

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.subheader('Heart Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 for Male, 0 for Female)')
    with col3:
        chestpain = st.text_input('Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic)')
    with col1:
        restingbp = st.text_input('Resting Blood Pressure')
    with col2:
        cholesterol = st.text_input('Cholesterol Level')
    with col3:
        fastingbloodsugar = st.text_input('Fasting Blood Sugar (1 if > 120 mg/dl, 0 otherwise)')
    with col1:
        restingecg = st.text_input('Resting Electrocardiographic Results (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)')
    with col2:
        maxheartrate = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exerciseangina = st.text_input('Exercise Induced Angina (1 for Yes, 0 for No)')
    with col1:
        oldpeak = st.text_input('Depression Induced by Exercise Relative to Rest')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy')
    with col1:
        thalassemia = st.text_input('Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)')

    heart_diagnosis = ''
    if st.button('Predict Heart Disease'):
        user_input = [age, sex, chestpain, restingbp, cholesterol, fastingbloodsugar, restingecg,
                      maxheartrate, exerciseangina, oldpeak, slope, ca, thalassemia]

        try:
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'

            # Save the result to a file
            result_file = r"C:\Users\burgu\Downloads\prediction_result.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_file, 'a') as file:
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Input Data: {user_input}\n")
                file.write(f"Prediction Result: {heart_diagnosis}\n\n")

        except ValueError:
            heart_diagnosis = 'Please enter valid numerical values for all fields'

    st.success(heart_diagnosis)

# Parkinson's Disease Prediction
elif selected == 'Parkinson\'s Prediction':
    st.subheader('Parkinson\'s Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo_Hz = st.text_input('MDVP:Fo(Hz)')
    with col2:
        MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz)')
    with col1:
        MDVP_Jitter = st.text_input('MDVP:Jitter(%)')
    with col2:
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
        MDVP_RAP = st.text_input('MDVP:RAP')
    with col1:
        MDVP_PPQ = st.text_input('MDVP:PPQ')
    with col2:
        Jitter_DDP = st.text_input('Jitter:DDP')
    with col3:
        MDVP_Shim = st.text_input('MDVP:Shimmer')
    with col1:
        MDVP_Shim_dB = st.text_input('MDVP:Shimmer(dB)')
    with col2:
        Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
    with col3:
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
    with col1:
        MDVP_APQ = st.text_input('MDVP:APQ')
    with col2:
        Shimmer_DDA = st.text_input('Shimmer:DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col1:
        spread1 = st.text_input('spread1')
    with col2:
        spread2 = st.text_input('spread2')
    with col3:
        D2 = st.text_input('D2')
    with col1:
        PPE = st.text_input('PPE')

    # Initialize parkinsons_diagnosis before try-except block
    parkinsons_diagnosis = None

    if st.button('Parkinson\'s Test Result'):
        user_input = [MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs,
                      MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim, MDVP_Shim_dB, Shimmer_APQ3, 
                      Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        try:
            # Convert all inputs to float, check if the input values are valid
            user_input = [float(x) for x in user_input]
            # Ensure that 'parkinsons_model' is properly loaded and ready for prediction
            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person has Parkinson\'s Disease'
            else:
                parkinsons_diagnosis = 'The person does not have Parkinson\'s Disease'

            # Save the result to a file
            result_file = r"C:\Users\burgu\Downloads\prediction_result.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_file, 'a') as file:
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Input Data: {user_input}\n")
                file.write(f"Prediction Result: {parkinsons_diagnosis}\n\n")

        except ValueError:
            # If input data is not valid, set a default error message
            parkinsons_diagnosis = 'Please enter valid numerical values for all fields'

    # Ensure that parkinsons_diagnosis is always defined before being displayed
    if parkinsons_diagnosis is not None:
        st.success(parkinsons_diagnosis)
    else:
        st.error("An error occurred while processing the input. Please check the values and try again.")

