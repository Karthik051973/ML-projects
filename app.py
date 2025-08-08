import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split

# Load dataset
car_dataset = pd.read_csv('car data.csv')

# Encode categorical features
car_dataset['Fuel_Type'] = car_dataset['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
car_dataset['Seller_Type'] = car_dataset['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
car_dataset['Transmission'] = car_dataset['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Prepare features and target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Train models
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

st.title("Car Selling Price Prediction")

st.sidebar.header("Input Features")
def user_input_features():
    year = st.sidebar.slider('Year', int(car_dataset['Year'].min()), int(car_dataset['Year'].max()), int(car_dataset['Year'].median()))
    present_price = st.sidebar.slider('Present Price (in lakhs)', float(car_dataset['Present_Price'].min()), float(car_dataset['Present_Price'].max()), float(car_dataset['Present_Price'].median()))
    kms_driven = st.sidebar.slider('Kms Driven', int(car_dataset['Kms_Driven'].min()), int(car_dataset['Kms_Driven'].max()), int(car_dataset['Kms_Driven'].median()))
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.sidebar.slider('Number of Previous Owners', int(car_dataset['Owner'].min()), int(car_dataset['Owner'].max()), int(car_dataset['Owner'].median()))

    fuel_type_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_type_dict = {'Dealer': 0, 'Individual': 1}
    transmission_dict = {'Manual': 0, 'Automatic': 1}
    data = {
        'Year': year,
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_type_dict[fuel_type],
        'Seller_Type': seller_type_dict[seller_type],
        'Transmission': transmission_dict[transmission],
        'Owner': owner
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

lin_prediction = lin_reg_model.predict(input_df)[0]
lasso_prediction = lass_reg_model.predict(input_df)[0]

st.subheader("Predicted Selling Prices")
st.write(f"Linear Regression Prediction: ₹{lin_prediction:.2f} lakhs")
st.write(f"Lasso Regression Prediction: ₹{lasso_prediction:.2f} lakhs")
