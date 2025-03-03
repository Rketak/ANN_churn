import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import seaborn as sns

## Load the model
model = tf.keras.models.load_model('model.keras')

## load the encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit UI Improvements
st.set_page_config(page_title='Customer Churn Prediction', layout='centered')
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        h1 {color: #0073e6; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.title('📊 Customer Churn Prediction')
st.write("Fill in customer details below to predict churn probability.")

# User Input Section
st.sidebar.header("🔹 Input Customer Details")
geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('🚻 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92, 30)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0, step=500.0)
credit_score = st.sidebar.number_input('📈 Credit Score', min_value=300, max_value=900, step=10)
estimated_salary = st.sidebar.number_input('💵 Estimated Salary', min_value=0.0, step=1000.0)
tenure = st.sidebar.slider('📆 Tenure (Years)', 0, 10, 5)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card?', ["No", "Yes"])
is_active_member = st.sidebar.selectbox('⚡ Is Active Member?', ["No", "Yes"])

# Data Preprocessing
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display results
st.subheader("📌 Churn Prediction Result")
st.write(f'🔍 **Churn Probability:** {prediction_proba:.2%}')

if prediction_proba > 0.5:
    st.error('🚨 The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')

# Visualization Section
st.subheader("📊 Data Visualization")
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=['Not Churn', 'Churn'], y=[1 - prediction_proba, prediction_proba], palette=['green', 'red'], ax=ax)
ax.set_ylabel("Probability")
ax.set_title("Churn Probability Distribution")
st.pyplot(fig)

# Add some visual appeal
st.markdown("""
    ---
    **🔹 About:**
    - This tool predicts customer churn probability using machine learning.
    - Adjust parameters in the sidebar to see how they impact the result.
""", unsafe_allow_html=True)
