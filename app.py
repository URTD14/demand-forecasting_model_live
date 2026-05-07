import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Demand Forecasting Dashboard")
st.write("Enter product details to predict demand.")

@st.cache_resource
def load_artifacts():
    with open('demand_forecasting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

# User Inputs
price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.slider("Discount (%)", 0, 100, 10)
inventory = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", [0, 1], help="0 = No, 1 = Yes")
comp_price = st.number_input("Competitor Pricing", min_value=0.0, value=55.0)
category = st.selectbox("Category", encoders['Category'].classes_)

if st.button("Predict Demand"):
    # Encode Category
    cat_encoded = encoders['Category'].transform([category])[0]
    
    # Features must match the training order: Price, Discount, Inventory, Promotion, Competitor, Category
    input_data = np.array([[price, discount, inventory, promotion, comp_price, cat_encoded]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Demand: {round(float(prediction[0]), 2)}")
