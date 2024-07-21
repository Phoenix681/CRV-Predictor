import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

# Load the model
with gzip.open('car_resale_model.pkl.gz', 'rb') as file:
    model = pickle.load(file)

st.title("Car Value Prediction")

st.write("### Enter the details of the car")

registered_year = st.number_input("Registered Year", min_value=1980, max_value=2024, value=2015)
engine_capacity = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1497)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=41200)
max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=500, value=117)
seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0)

insurance = st.selectbox("Insurance Type", ['Comprehensive', 'Third Party', 'Zero Dep'])
transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
owner_type = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner', 'Fifth Owner'])
fuel_type = st.selectbox("Fuel Type", ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'])
body_type = st.selectbox("Body Type", ['Convertibles', 'Coupe', 'Hatchback', 'MUV', 'Minivans', 'Pickup', 'SUV', 'Sedan', 'Wagon'])
city = st.selectbox("City", ['Agra', 'Ahmedabad', 'Bangalore', 'Chandigarh', 'Chennai', 'Delhi', 'Gurgaon', 'Hyderabad', 'Jaipur', 'Kolkata', 'Lucknow', 'Mumbai', 'Pune'])
company = st.selectbox("Company", ['Audi', 'BMW', 'Bentley', 'Chevrolet', 'Citroen', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Hindustan', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'OpelCorsa', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'])

insurance_mapping = {'Comprehensive': [1, 0, 0], 'Third Party': [0, 1, 0,], 'Zero Dep': [0, 0, 1]}
transmission_mapping = {'Automatic': [1, 0], 'Manual': [0, 1]}
owner_mapping = {'First Owner': [1, 0, 0, 0, 0], 'Second Owner': [0, 1, 0, 0, 0], 'Third Owner': [0, 0, 1, 0, 0], 'Fourth Owner': [0, 0, 0, 1, 0], 'Fifth Owner': [0, 0, 0, 0, 1]}
fuel_mapping = {'CNG': [1, 0, 0, 0, 0], 'Diesel': [0, 1, 0, 0, 0], 'Electric': [0, 0, 1, 0, 0], 'LPG': [0, 0, 0, 1, 0], 'Petrol': [0, 0, 0, 0, 1]}
body_mapping = {'Convertibles': [1, 0, 0, 0, 0, 0, 0, 0, 0], 'Coupe': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'Hatchback': [0, 0, 1, 0, 0, 0, 0, 0, 0], 'MUV': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'Minivans': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'Pickup': [0, 0, 0, 0, 0, 1, 0, 0, 0], 'SUV': [0, 0, 0, 0, 0, 0, 1, 0, 0], 'Sedan': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'Wagon': [0, 0, 0, 0, 0, 0, 0, 0, 1]}
city_mapping = {'Agra': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Ahmedabad': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bangalore': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Chandigarh': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Chennai': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'Delhi': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Gurgaon': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'Jaipur': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Kolkata': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Lucknow': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Mumbai': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Pune': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
company_mapping = {'Audi': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'BMW': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bentley': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Chevrolet': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Citroen': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Daewoo': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Datsun': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Fiat': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Force': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Ford': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Hindustan': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Honda': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Hyundai': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Isuzu': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Jaguar': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Jeep': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Kia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Land': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Lexus': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'MG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Mahindra': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Maruti': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Mercedes': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Mini': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Mitsubishi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Nissan': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'Opel': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Porsche': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Renault': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'Skoda': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Tata': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Toyota': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Volkswagen': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Volvo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

def predict_price(registered_year, engine_capacity, kms_driven, max_power, seats, mileage, insurance, transmission, owner_type, fuel_type, body_type, city, company):
    insurance_encoded = insurance_mapping[insurance]
    transmission_encoded = transmission_mapping[transmission]
    owner_encoded = owner_mapping[owner_type]
    fuel_encoded = fuel_mapping[fuel_type]
    body_encoded = body_mapping[body_type]
    city_encoded = city_mapping[city]
    company_encoded = company_mapping[company]
    
    features = np.array([
        registered_year, engine_capacity, kms_driven, max_power, seats, mileage,
        *insurance_encoded, *transmission_encoded, *owner_encoded, *fuel_encoded,
        *body_encoded, *city_encoded, *company_encoded
    ]).reshape(1, -1)
    
    predicted_price = model.predict(features)[0]
    return predicted_price

if st.button("Predict"):
    predicted_price = predict_price(registered_year, engine_capacity, kms_driven, max_power, seats, mileage, insurance, transmission, owner_type, fuel_type, body_type, city, company)
    st.write(f"Predicted Car Value: ₹ {predicted_price:.2f} ")
