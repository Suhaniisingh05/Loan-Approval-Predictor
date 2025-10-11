import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Configuration & Model Loading ---

# Load the saved model and column list
try:
    with open('loan_prediction_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    
    model = model_data['model']
    # The 'columns' list tells us the exact order and names of the 
    # features the model was trained on (e.g., Gender_Male, Married_Yes)
    model_columns = model_data['columns'] 
    
except FileNotFoundError:
    st.error("Model file not found. Please run 'model_trainer.py' first.")
    st.stop()

# Set up the Streamlit interface
st.set_page_config(page_title="Financial Risk Predictor", layout="centered")
st.title("🏦 Automated Loan Eligibility Checker")
st.markdown("---")


# --- User Input Fields ---

st.header("1. Applicant Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
with col2:
    married = st.selectbox('Married', ['Yes', 'No'])
with col3:
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])


col4, col5 = st.columns(2)

with col4:
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
with col5:
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])

# --- Financial Inputs ---

st.header("2. Financial & Loan Request")

col6, col7 = st.columns(2)

with col6:
    # Use standard 1.0 (Good Credit) as the key predictor
    credit_history = st.radio('Credit History (1.0 = Met Guidelines)', [1.0, 0.0])

with col7:
    loan_amount = st.slider('Loan Amount (in thousands)', min_value=10, max_value=700, value=150, step=10)


st.subheader("Income Details")
col8, col9 = st.columns(2)

with col8:
    applicant_income = st.number_input('Applicant Income', min_value=0, value=4000)

with col9:
    coapplicant_income = st.number_input('Co-Applicant Income', min_value=0, value=1000)


property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])
loan_term = st.selectbox('Loan Term (Months)', [360.0, 180.0, 60.0, 480.0])


# --- Prediction Logic (CORRECTED) ---

if st.button('Predict Loan Status', help="Click to see the model's prediction"):
    
    # 1. Prepare Features and Handle Encoding/Feature Engineering
    
    # Feature Engineering: Total Income
    total_income = applicant_income + coapplicant_income
    
    # Base structure to hold all 0/1 values for the model
    # We initialize all feature columns that the model expects to zero
    final_features = pd.DataFrame(0, index=[0], columns=model_columns)

    # 2. Map User Inputs to the Model's EXPECTED Format (One-Hot Encoded)

    # Numerical Features (Set directly)
    final_features['Total_Income'] = total_income
    final_features['LoanAmount'] = loan_amount
    final_features['Loan_Amount_Term'] = loan_term
    final_features['Credit_History'] = credit_history
    
    # Categorical Features (Set '1' for the selected category)
    # The columns are named as [Original_Feature]_[Category_Value]
    
    if gender == 'Male':
        final_features['Gender_Male'] = 1
        
    if married == 'Yes':
        final_features['Married_Yes'] = 1
        
    if education == 'Not Graduate':
        final_features['Education_Not Graduate'] = 1
        
    if self_employed == 'Yes':
        final_features['Self_Employed_Yes'] = 1
        
    # Dependents (Note the columns match 'Dependents_1', 'Dependents_2', etc.)
    if dependents != '0': # If it's 1, 2, or 3+, set the corresponding column to 1
        final_features[f'Dependents_{dependents}'] = 1

    # Property Area
    if property_area == 'Semiurban':
        final_features['Property_Area_Semiurban'] = 1
    elif property_area == 'Urban':
        final_features['Property_Area_Urban'] = 1
    # 'Rural' is the reference category (where both Urban/Semiurban are 0)
    
    
    # --- Make Prediction ---
    
    # The input_df (final_features) now has the exact column names and order as the model
    prediction = model.predict(final_features)[0]
    prediction_proba = model.predict_proba(final_features)[0][1] 

    st.markdown("---")
    st.subheader("Model Decision:")

    if prediction == 1:
        st.success(f"**Loan Status: APPROVED!**")
        st.info(f"Model Confidence: {prediction_proba*100:.2f}%")
        st.balloons()
    else:
        st.error(f"**Loan Status: REJECTED.**")
        st.warning(f"Model Confidence: {(1 - prediction_proba)*100:.2f}%")
        st.info("Recommendation: The bank should review the application for risk factors.")