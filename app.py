#app.py
import streamlit as st
import pickle
import numpy as np

#load model and scaler
with open('knn_model.pkl', 'rb')as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb')as f:
    scaler = pickle.load(f)

#Title and description
st.set_page_config(page_title="Loan approval Predictor",page_icon='💰')
st.title("💰Loan Approval Prediction")
st.markdown("Enter your financial details below to see if your loan will be approved.")        

# input field in two column for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    annual_income = st.slider("Annual Income (in $1,000s)", 20, 200, 60)
    credit_score = st.slider("Credit score", 300, 850, 650)
    loan_amount = st.slider("Loan Amount(in $1,000s)", 5, 100, 30)

with col2:
    employment_year = st.slider("Year of Employment",0, 40, 5) 
    debt_to_income = st.slider("Debt-Income Ratio",0.0,1.0,0.3, step=0.01)
    dependents = st.slider("Number of Dependents", 0, 5, 1)

#Prediction Button
if st.button("Predict laon Approval", type="primary"):
    #Prepare input as 2D array
    input_data = np.array([[age, annual_income, credit_score, loan_amount,
                           employment_year, debt_to_income, dependents]])

#scale the input
    input_scaled = scaler.transform(input_data)           

#predict class and probabilities
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

#display result
    st.divider()
    if prediction == 1:
         st.success(f"Loan Approved! (confidence: {proba[1]*100:.1f}%)")
    else:
        st.error(f"Loan Rejected. (confidence: {proba[0]*100:.1f}%)")

#Optional show probability bar
    st.progress(proba[1],text=f"Approval Probability: {proba[1]*100:.1f}%")  
