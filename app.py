
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

# Define the feature names from X.columns (from your notebook state)
feature_names = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction App')
st.write('Enter the feature values below to predict delivery delay.')

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}:', value=0.0)

# Predict button
if st.button('Predict Delivery Delay'):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Delivery is likely to be delayed!')
    else:
        st.success('Delivery is likely to be on time.')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
