import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the DataFrame
df = pd.read_csv("df.csv")

# Load the model using joblib
pipe = joblib.load("Prediction_model.joblib")

# Streamlit app title
st.title("Laptop Price Predictor")

# User inputs
Company_Name = st.selectbox('Company', df['Company'].unique())
Laptop_Type = st.selectbox("Laptop Type", df['TypeName'].unique())
RAM = st.selectbox("Ram(in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
Weight = st.number_input("Weight(in kg)")
Touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])
IPS = st.selectbox("IPS", ['No', 'Yes'])
Size_of_screen = st.number_input('Screen Size')
Resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3200x1800', '2880x1800', '2560x1600', '2304x1440'])
CPU = st.selectbox('CPU', df['Cpu brand'].unique())
Harddisk = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
SSD = st.selectbox('SSD(in GB)', [128, 256, 512, 1024])
GPU = st.selectbox('Graphic card', df['Gpu brand'].unique())
OperatingSys = st.selectbox('Operating system', df['os'].unique())

# Prediction button
if st.button('Predict'):
    # Convert categorical inputs to numerical values
    if Touchscreen == "Yes":
        Touchscreen = 1
    else:
        Touchscreen = 0
    if IPS == "Yes":
        IPS = 1
    else:
        IPS = 0

    # Calculate pixels per inch (ppi)
    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Size_of_screen

    # Create input query for the model
    query = np.array([Company_Name, Laptop_Type, RAM, Weight,
                      Touchscreen, IPS, ppi, CPU, Harddisk, SSD, GPU, OperatingSys])
    query = query.reshape(1, 12)

    # Predict the price using the model
    prediction = str(int(np.exp(pipe.predict(query)[0])))

    # Display the prediction
    st.title("Predicted Price of Laptop is $ " + prediction)
