import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('car data.csv')

car_dataset = load_data()

# Encode categorical variables
car_dataset['Fuel_Type'] = car_dataset['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
car_dataset['Seller_Type'] = car_dataset['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
car_dataset['Transmission'] = car_dataset['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Features and target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Title
st.title("🚗 Car Selling Price Prediction")

# Feedback Button at top
feedback_url = "https://docs.google.com/forms/d/e/1FAIpQLScAF-TnTBnDlmeOUVxjeyZaadMCuh9lWHSzqMgInEPWq_YZOw/viewform"
st.markdown(f"<a href='{feedback_url}' target='_blank'><button style='background-color: #4CAF50; color: white; padding: 8px 16px; font-size: 16px; border: none; border-radius: 5px;'>📝 Give Feedback</button></a>", unsafe_allow_html=True)

# Frontend form
st.header("Enter Car Details")
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input('Year', min_value=int(car_dataset['Year'].min()), max_value=int(car_dataset['Year'].max()), value=int(car_dataset['Year'].median()))
    kms_driven = st.number_input('Kms Driven', min_value=int(car_dataset['Kms_Driven'].min()), max_value=int(car_dataset['Kms_Driven'].max()), value=int(car_dataset['Kms_Driven'].median()))
    owner = st.number_input('Number of Previous Owners', min_value=int(car_dataset['Owner'].min()), max_value=int(car_dataset['Owner'].max()), value=int(car_dataset['Owner'].median()))

with col2:
    present_price = st.number_input('Present Price (in lakhs)', min_value=float(car_dataset['Present_Price'].min()), max_value=float(car_dataset['Present_Price'].max()), value=float(car_dataset['Present_Price'].median()))
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])

with col3:
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

# Convert inputs to DataFrame
fuel_type_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_type_dict = {'Dealer': 0, 'Individual': 1}
transmission_dict = {'Manual': 0, 'Automatic': 1}

input_df = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Fuel_Type': [fuel_type_dict[fuel_type]],
    'Seller_Type': [seller_type_dict[seller_type]],
    'Transmission': [transmission_dict[transmission]],
    'Owner': [owner]
})

# Predict when button clicked
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Selling Price")
    st.success(f"₹{prediction:.2f} lakhs")

