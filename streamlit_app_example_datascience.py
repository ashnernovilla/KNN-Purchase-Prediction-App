# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 10:47:44 2025

@author: ASHNER_NOVILLA
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_model():
    # Load data
    data = pd.read_csv("https://raw.githubusercontent.com/sakshi2k/Social_Network_Ads/refs/heads/master/Social_Network_Ads.csv")

    # Drop User ID
    data.drop(columns=['User ID'], inplace=True)

    # Encode Gender
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])  # Female=0, Male=1

    # Features and target
    X = data[['Gender', 'Age', 'EstimatedSalary']]
    y = data['Purchased']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)

    return model, scaler, le

# Load model and scaler
model, scaler, le = load_model()

st.title("K-Nearest Neighbor Prediction App")
st.write("Predict if a user will purchase a product based on age and estimated salary.")

# Inputs
gender = st.selectbox("Select Gender", ['Female', 'Male'])
age = st.slider("Select Age", 18, 60, 30)
salary = st.number_input("Estimated Salary", min_value=10000, max_value=150000, value=50000)

if st.button("Predict"):
   gender_encoded = le.transform([gender])[0]  # 0 for Female, 1 for Male
   input_data = np.array([[gender_encoded, age, salary]])
   input_scaled = scaler.transform(input_data)
   prediction = model.predict(input_scaled)[0]

   if prediction == 1:
       st.markdown(f"<h3 style='color:green;'>✅ Prediction: Will Purchase</h3>", unsafe_allow_html=True)
   else:
       st.markdown(f"<h3 style='color:red;'>❌ Prediction: Will Not Purchase</h3>", unsafe_allow_html=True)