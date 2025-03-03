# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# ## Load the model
# model = tf.keras.models.load_model('regression_model.keras')

# ## load the encoder and scaler
# with open('label_encoder_gender.pkl','rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl','rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl','rb') as file:
#     scaler = pickle.load(file)

# st.sidebar.header("🔹 Input Customer Details")
# geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
# gender = st.sidebar.selectbox('🚻 Gender', label_encoder_gender.classes_)
# age = st.sidebar.slider('🎂 Age', 18, 92, 30)
# balance = st.sidebar.number_input('💰 Balance', min_value=0.0, step=500.0)
# credit_score = st.sidebar.number_input('📈 Credit Score', min_value=300, max_value=900, step=10)
# exited = st.sidebar.selectbox('Exited',[0,1])
# tenure = st.sidebar.slider('📆 Tenure (Years)', 0, 10, 5)
# num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4, 1)
# has_cr_card = st.sidebar.selectbox('💳 Has Credit Card?', ["No", "Yes"])
# is_active_member = st.sidebar.selectbox('⚡ Is Active Member?', ["No", "Yes"])

# # Data Preprocessing
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [1 if has_cr_card == "Yes" else 0],
#     'IsActiveMember': [1 if is_active_member == "Yes" else 0],
#     'Exited': [exited]
# })

# # One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Prediction
# prediction = model.predict(input_data_scaled)
# prediction_salary = prediction[0][0]

# st.write(f'🔍 **Estimed Salary: ${prediction_salary:.2f}')

## 2

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the model
# model = tf.keras.models.load_model('regression_model.keras')

# # Load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# # Set page title and layout
# st.set_page_config(page_title="Customer Salary Prediction", page_icon="💰", layout="wide")

# # Title and Description
# st.markdown("<h1 style='text-align: center;'>💼 Customer Salary Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: grey;'>Enter customer details to predict their estimated salary.</p>", unsafe_allow_html=True)
# st.divider()

# # Sidebar for User Inputs
# st.sidebar.header("🔹 Enter Customer Details")

# # Create columns for better UI
# col1, col2 = st.columns(2)

# with col1:
#     geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
#     gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
#     age = st.slider('🎂 Age', 18, 92, 30)
#     balance = st.number_input('💰 Balance', min_value=0.0, step=500.0)
#     credit_score = st.number_input('📈 Credit Score', min_value=300, max_value=900, step=10)

# with col2:
#     tenure = st.slider('📆 Tenure (Years)', 0, 10, 5)
#     num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
#     has_cr_card = st.selectbox('💳 Has Credit Card?', ["No", "Yes"])
#     is_active_member = st.selectbox('⚡ Is Active Member?', ["No", "Yes"])
#     exited = st.selectbox('❌ Customer Exited?', [0, 1])

# # Preprocess input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [1 if has_cr_card == "Yes" else 0],
#     'IsActiveMember': [1 if is_active_member == "Yes" else 0],
#     'Exited': [exited]
# })

# # One-hot encode 'Geography'
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Prediction
# prediction = model.predict(input_data_scaled)
# prediction_salary = prediction[0][0]

# # Display Results
# st.divider()
# st.markdown("<h2 style='text-align: center;'>📊 Estimated Salary</h2>", unsafe_allow_html=True)
# st.markdown(f"<h3 style='text-align: center; color: green;'>💵 ${prediction_salary:.2f}</h3>", unsafe_allow_html=True)
# st.success("✅ Prediction Complete! The estimated salary has been calculated based on the input data.")

## 3

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model('regression_model.keras')

# Load encoders & scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set up Streamlit UI
st.set_page_config(page_title="Customer Salary Prediction", page_icon="💰", layout="wide")
st.markdown("<h1 style='text-align: center;'>💼 Customer Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter customer details to predict their estimated salary.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar for User Inputs
st.sidebar.header("🔹 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    balance = st.number_input('💰 Balance', min_value=0.0, step=500.0)
    credit_score = st.number_input('📈 Credit Score', min_value=300, max_value=900, step=10)

with col2:
    tenure = st.slider('📆 Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', ["No", "Yes"])
    is_active_member = st.selectbox('⚡ Is Active Member?', ["No", "Yes"])
    exited = st.selectbox('❌ Customer Exited?', [0, 1])

# Preprocess Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'Exited': [exited]
})

# One-hot encoding 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct feature order
expected_columns = scaler.feature_names_in_
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)

# Ensure correct indexing for prediction
prediction_salary = float(prediction[0])  # Works for both 1D & 2D output

# Display Results
st.divider()
st.markdown("<h2 style='text-align: center;'>📊 Estimated Salary</h2>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: green;'>💵 ${prediction_salary:.2f}</h3>", unsafe_allow_html=True)
st.success("✅ Prediction Complete! The estimated salary has been calculated based on the input data.")
