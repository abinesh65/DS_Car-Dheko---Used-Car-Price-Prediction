import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing steps
model = joblib.load('C:/Users/91894/OneDrive/Desktop/cardekho/car_price_prediction_model.pkl')
label_encoders = joblib.load('C:/Users/91894/OneDrive/Desktop/cardekho/label_encoders.pkl')
scalers = joblib.load('C:/Users/91894/OneDrive/Desktop/cardekho/scalers.pkl')
data = pd.read_csv('C:/Users/91894/OneDrive/Desktop/cardekho/car_dekho_cleaned_dataset_Raw.csv', low_memory=False)


# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    df['brand_popularity'] = df['oem'].map(data.groupby('oem')['price'].mean().to_dict())
    df['mileage_normalized'] = df['mileage'] / df['car_age']
    
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)

    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])
    
    df.fillna(0, inplace=True)
    return df

st.title("Car Price Prediction")
st.sidebar.header('Input Car Features')

def visual_selectbox(label, options, index=0):
    return st.sidebar.selectbox(label, options, index=index)

selected_oem = visual_selectbox('1. OEM', data['oem'].unique())
filtered_data = filter_data(oem=selected_oem)
selected_model = visual_selectbox('2. Model', filtered_data['model'].unique())
filtered_data = filter_data(oem=selected_oem, model=selected_model)
body_type = visual_selectbox('3. Body Type', filtered_data['bt'].unique())
fuel_type = visual_selectbox('4. Fuel Type', filtered_data['ft'].unique())
transmission = visual_selectbox('5. Transmission', filtered_data['transmission'].unique())
seat_count = visual_selectbox('6. Seats', filtered_data['Seats'].unique())
selected_variant = visual_selectbox('7. Variant', filtered_data['variantName'].unique())
modelYear = st.sidebar.number_input('8. Year', 1980, 2024, 2015)
ownerNo = st.sidebar.number_input('9. Owners', 0, 10, 1)
km = st.sidebar.number_input('10. Km Driven', 0, 500000, 10000)
mileage = st.sidebar.slider('11. Mileage (kmpl)', float(filtered_data['mileage'].min()), float(filtered_data['mileage'].max()), step=0.5)
city = visual_selectbox('12. City', data['City'].unique())

user_df = pd.DataFrame({
    'ft': [fuel_type],
    'bt': [body_type],
    'km': [km],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'oem': [selected_oem],
    'model': [selected_model],
    'modelYear': [modelYear],
    'variantName': [selected_variant],
    'City': [city],
    'mileage': [mileage],
    'Seats': [seat_count],
    'car_age': [2024 - modelYear],
    'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem, 0)],
    'mileage_normalized': [mileage / (2024 - modelYear)]
})

user_df = user_df[features]
user_df = preprocess_input(user_df)

if st.sidebar.button('Predict'):
    try:
        prediction = model.predict(user_df)
        st.markdown(f"""<div style='text-align:center;'>
            <h2 style='color:maroon;'>Predicted Price</h2>
            <p style='font-size:36px; font-weight:bold;'>₹{prediction[0]:,.2f}</p>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")
