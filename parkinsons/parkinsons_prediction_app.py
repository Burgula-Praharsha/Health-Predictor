import os
import pickle  # pre-trained model loading
import streamlit as st  # web app
from streamlit_option_menu import option_menu
import datetime  # Import datetime for timestamp

st.set_page_config(page_title='Parkinson\'s Disease Prediction',
                   layout='wide',
                   page_icon="🧠")

# Load Parkinson's Disease model
parkinsons_model = pickle.load(open(r"C:\Users\burgu\Downloads\parkinsons_model.sav", 'rb'))

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Parkinson\'s Disease Prediction'],
                           menu_icon='hospital-fill', icons=['brain'], default_index=0)

if selected == 'Parkinson\'s Disease Prediction':
    st.subheader('Parkinson\'s Disease Prediction using Machine Learning')
    
    col1, col2, col3 = st.columns(3)
    
    # Input fields for all the features
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
                
            # Optional: Save the result to a file with timestamp
            result_file = r"C:\Users\burgu\Downloads\parkinsons_prediction_results.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_file, 'a') as file:
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Input Data: {user_input}\n")
                file.write(f"Prediction Result: {parkinsons_diagnosis}\n\n")
            
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numerical values for all fields'
            
st.success(parkinsons_diagnosis)
