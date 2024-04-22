import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib

# Load the XGBoost model
classifier1 = joblib.load('xgb_regressor.pkl')
# classifier2 = joblib.load('random_forest_regressor.pkl')
classifier3 = keras.models.load_model("trained_model.h5")

def draw_donut_chart(data, title):
    if(data<0):
        data =0
    sizes = [data,100-data]

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=['Unsafe','Safe'], colors=['red','green'], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))

    # Draw a circle in the middle to make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.7, color='white', fc='white', linewidth=1.25)
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Set title
    ax.set_title(title, size=16)

    # Set background color
    ax.set_facecolor('lightgrey')

    st.pyplot(fig)

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

    # Prepare features for prediction
    features = [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, Age, Diabetic, PhysicalActivity, GenHealth_encoding, SleepTime, Asthma, KidneyDisease, SkinCancer]
    
    # Reshape features into 2D array for prediction
    features_arr_float = np.array(features, dtype=float).reshape(1, -1)

    # Make prediction
    prediction1 = classifier1.predict(features_arr_float)
    # prediction2 = classifier2.predict(features_arr_object)
    prediction3 = classifier3.predict(features_arr_float)[0]
    print(prediction1)
    print(prediction3)
    return (prediction1  + prediction3)/2

def main():
    st.title("Cardiac Arrest Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cardiac Arrest Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # Input fields for user input
    BMI = st.slider('BMI:', min_value=0.0, max_value=100.0, value=28.32,step=0.0001)
    # weight = st.text_input("Weight (Kg)", value = 50)
    # height = st.text_input("Height (cm)", value = 165)

    # Convert inputs to float
    # weight = float(weight)
    # height = float(height) / 100  # converting cm to meters

    # Calculate BMI
    # BMI = weight / (height * height)
    Smoking = st.radio("Smoking", ("Yes", "No"))
    AlcoholDrinking = st.radio("Alcohol Drinking", ("Yes", "No"))
    Stroke = st.radio("Stroke", ("Yes", "No"))
    # PhysicalHealth = st.text_input("Physical Health", "Type Here")
    PhysicalHealth = st.slider('Physical Health:', min_value=0, max_value=100, value=0)

    # MentalHealth = st.text_input("Mental Health", "Type Here")
    MentalHealth = st.slider('Mental Health:', min_value=0, max_value=100, value=0)

    DiffWalking = st.radio("Difficulty Walking", ("Yes", "No"))
    Sex = st.radio("Sex", ("Male", "Female"))
    # Age = st.text_input("Age", "Type Here")
    Age = st.slider('Age:', min_value=1, max_value=80, value=54)

    Diabetic = st.radio("Diabetic", ("Yes", "No"))
    PhysicalActivity = st.radio("Physical Activity", ("Yes", "No"))
    GenHealth = st.radio("General Health", ("Excellent", "Very good", "Good", "Fair", "Poor"))
    # SleepTime = st.text_input("Sleep Time", "Type Here")
    SleepTime = st.slider('Sleep Time:', min_value=0.0, max_value=24.0, value=7.0,step=0.5)

    Asthma = st.radio("Asthma", ("Yes", "No"))
    KidneyDisease = st.radio("Kidney Disease", ("Yes", "No"))
    SkinCancer = st.radio("Skin Cancer", ("Yes", "No"))
    
    result = []
    if st.button("Predict"):
        check=True
        result = predict_cardiac_arrest(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, Age, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer)
    # st.success('The predicted class is {}'.format(result))
    # if st.button("About"):
    #     st.text("Cardiac Arrest Prediction App")
    if len(result) and check > 0:
        result = [value * 100 for value in result]
        draw_donut_chart(result[0], "Cardiac Arrest Prediction Percentage")

if __name__=='__main__':
    main()
