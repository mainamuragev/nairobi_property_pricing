import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Load the original data to get available locations
df = pd.read_csv('cleaned_properties.csv')
locations = sorted(df['location'].dropna().unique())

# MAE from your best model (Random Forest)
BEST_MAE = 4_801_158

st.set_page_config(page_title="Nairobi Property Price Predictor", layout="centered")
st.title("🏠 Nairobi Property Price Estimator")
st.markdown("Enter property details to get an estimated price range.")

# Input widgets
location = st.selectbox("Location", locations)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2, step=1)

if st.button("Estimate Price"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'location': [location],
        'bedrooms': [bedrooms]
    })
    
    # Predict
    pred = model.predict(input_data)[0]
    lower = max(pred - BEST_MAE, 0)  # ensure non-negative
    upper = pred + BEST_MAE
    
    st.success(f"**Estimated Price:** KES {pred:,.0f}")
    st.info(f"**Typical Range:** KES {lower:,.0f} – KES {upper:,.0f}")
    
    # Explanation of top drivers
    st.markdown("---")
    st.markdown("**Top price drivers in this model:**")
    st.markdown("- Number of bedrooms (most important)")
    st.markdown("- Premium locations: Westlands, Loresho, areas along Kiambu Road")
    st.caption("Prediction based on location and bedroom count. Actual prices may vary.")
