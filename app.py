import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Load the classifier model
classifier = load_model("trained_model.h5")

def predict_cardiac_arrest(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, Age, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer):
    # Encode categorical variables as needed
    Smoking = 1 if Smoking == "Yes" else 0
    AlcoholDrinking = 1 if AlcoholDrinking == "Yes" else 0
    Stroke = 1 if Stroke == "Yes" else 0
    DiffWalking = 1 if DiffWalking == "Yes" else 0
    Sex = 1 if Sex == "Male" else 0  # Assuming Male is encoded as 1 and Female as 0
    Diabetic = 1 if Diabetic == "Yes" else 0
    PhysicalActivity = 1 if PhysicalActivity == "Yes" else 0
    GenHealth_encoding = 0 if GenHealth == "Excellent" else 4 if GenHealth == "Very good" else 2 if GenHealth == "Good" else 1 if GenHealth == "Fair" else 4
    Asthma = 1 if Asthma == "Yes" else 0
    KidneyDisease = 1 if KidneyDisease == "Yes" else 0
    SkinCancer = 1 if SkinCancer == "Yes" else 0
    
    # Print encoded values
    print("Smoking:", Smoking)
    print("Alcohol Drinking:", AlcoholDrinking)
    print("Stroke:", Stroke)
    print("Difficulty Walking:", DiffWalking)
    print("Sex:", Sex)
    print("Diabetic:", Diabetic)
    print("Physical Activity:", PhysicalActivity)
    print("General Health encoding:", GenHealth_encoding)
    print("Asthma:", Asthma)
    print("Kidney Disease:", KidneyDisease)
    print("Skin Cancer:", SkinCancer)
    
    # Prepare features for prediction
    features = [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, Age, Diabetic, PhysicalActivity, GenHealth_encoding, SleepTime, Asthma, KidneyDisease, SkinCancer]
    
    # Reshape features into 2D array for prediction
    features_arr = np.array(features, dtype=object).reshape(1, -1)

    # Make prediction
    prediction = classifier.predict(features_arr)
    
    return prediction[0]  # Assuming the prediction is a single value

def main():
    st.title("Cardiac Arrest Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cardiac Arrest Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # Input fields for user input
    BMI = st.text_input("BMI", "Type Here")
    Smoking = st.radio("Smoking", ("Yes", "No"))
    AlcoholDrinking = st.radio("Alcohol Drinking", ("Yes", "No"))
    Stroke = st.radio("Stroke", ("Yes", "No"))
    PhysicalHealth = st.text_input("Physical Health", "Type Here")
    MentalHealth = st.text_input("Mental Health", "Type Here")
    DiffWalking = st.radio("Difficulty Walking", ("Yes", "No"))
    Sex = st.radio("Sex", ("Male", "Female"))
    Age = st.text_input("Age", "Type Here")
    Diabetic = st.radio("Diabetic", ("Yes", "No"))
    PhysicalActivity = st.radio("Physical Activity", ("Yes", "No"))
    GenHealth = st.radio("General Health", ("Excellent", "Very good", "Good", "Fair", "Poor"))
    SleepTime = st.text_input("Sleep Time", "Type Here")
    Asthma = st.radio("Asthma", ("Yes", "No"))
    KidneyDisease = st.radio("Kidney Disease", ("Yes", "No"))
    SkinCancer = st.radio("Skin Cancer", ("Yes", "No"))
    
    result = ""
    if st.button("Predict"):
        result = predict_cardiac_arrest(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, Age, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer)
    st.success('The predicted class is {}'.format(result))
    if st.button("About"):
        st.text("Cardiac Arrest Prediction App")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
