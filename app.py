import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('Cardetails.csv')

# Helper function to get brand name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Clean the car name column
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Drop unnecessary columns
cars_data.drop(columns=['torque'], inplace=True)

# Define options for the form
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Ownership type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

# Replace categorical string values with numbers
def encode_input_data(input_data):
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 
                                 'Fourth & Above Owner', 'Test Drive Car'], 
                                [1, 2, 3, 4, 5], inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    
    # Handle car brands dynamically
    brand_mapping = {
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 
        'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12,
        'Mercedes-Benz': 13, 'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17,
        'Nissan': 18, 'Lexus': 19, 'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23,
        'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29, 
        'Isuzu': 30, 'Opel': 31
    }
    
    # If a new brand appears, assign it a unique code
    if input_data['name'].values[0] not in brand_mapping:
        brand_mapping[input_data['name'].values[0]] = len(brand_mapping) + 1
        
    input_data['name'].replace(brand_mapping, inplace=True)

    return input_data

if st.button("Predict"):
    # Prepare input data for model
    input_data_model = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
                                    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    # Encode input data
    input_data_model = encode_input_data(input_data_model)

    # Predict car price
    car_price = model.predict(input_data_model)

    st.markdown(f'Car Price is predicted to be ₹ {car_price[0]:,.2f}')
