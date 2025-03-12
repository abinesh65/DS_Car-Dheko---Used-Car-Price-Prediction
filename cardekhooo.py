import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessing steps
model = joblib.load("C:\\Users\\91894\\Downloads\\final_random_forest_model1.pkl")
label_encoders = joblib.load("C:\\Users\\91894\\Downloads\\label_encoder.pkl")
scalers = joblib.load("C:\\Users\\91894\\min_max.pkl")  # MinMaxScaler for inverse transformation

# Load dataset for filtering options
data = pd.read_csv('C:\\Users\\91894\\OneDrive\\Desktop\\cardekho\\cleaned_cardata.csv')

# Features used for training
features = ['Mileage', 'Model_year', 'Kilometer_Driven', 'Engine_displacement', 'Fuel_type', 'Model',
            'Transmission', 'Owner_No.', 'Body_type', 'City', 'Max_power', 'Car_Age', 'Mileage_normalized']

# Function to preprocess input data
def preprocess_input(df):
    df['Car_Age'] = 2024 - df['Model_year']
    df['Mileage_normalized'] = df['Mileage'] / np.where(df['Car_Age'] == 0, 1, df['Car_Age'])  # Avoid division by zero
    return df

# Streamlit UI
st.set_page_config(page_title="Car Price Prediction", page_icon=":red_car:", layout="wide")
st.title("🚗 Car Price Prediction")

# Sidebar Inputs
st.sidebar.header('Input Car Features')

# Car Image
car_image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRsQfjGnjd8gBWy8GX8MceHN-yLdABrM8uzw&s"
st.image(car_image_path, caption="Sample Car Image", use_container_width=True)


# UI Styling
st.markdown(
    """
    <style>
    .result-container {
        text-align: center;
        background-color: #FFF8E7;
        padding: 10px;
        border-radius: 10px;
        width: 70%;
        margin: 0 auto;
    }
    .prediction-title {
        font-size: 28px;
        color: maroon;
    }
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: maroon;
    }
    .info {
        font-size: 18px;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Inputs for User
selected_Model = st.sidebar.selectbox('1. Car Model', sorted(data['Model'].unique()))

filtered_data = data[data['Model'] == selected_Model]
Body_type = st.sidebar.selectbox('2. Body Type', sorted(filtered_data['Body_type'].unique()))
Fuel_type = st.sidebar.selectbox('3. Fuel Type', sorted(filtered_data['Fuel_type'].unique()))
Transmission = st.sidebar.selectbox('4. Transmission Type', sorted(filtered_data['Transmission'].unique()))
ModelYear = st.sidebar.number_input('5. Model Year', min_value=1980, max_value=2024, value=2015)
OwnerNo = st.sidebar.number_input('6. Number of Previous Owners', min_value=0, max_value=10, value=1)
KilometersDriven = st.sidebar.number_input('7. Kilometers Driven', min_value=0, max_value=500000, value=10000)

# Adjust mileage slider safely
min_mileage = np.floor(filtered_data['Mileage'].min())
max_mileage = np.ceil(filtered_data['Mileage'].max())

Mileage = st.sidebar.slider('8. Mileage (kmpl)', min_value=float(min_mileage), max_value=float(max_mileage), value=float(min_mileage), step=0.5)
City = st.sidebar.selectbox('9. City', sorted(data['City'].unique()))
Max_power = st.sidebar.number_input('10. Max Power (bhp)', min_value=0, max_value=1000, value=100)
Engine_displacement = st.sidebar.number_input('11. Engine Displacement (cc)', min_value=0, max_value=10000, value=1000)

# Create DataFrame for User Input
user_input_data = {
    'Fuel_type': [Fuel_type],
    'Body_type': [Body_type],
    'Kilometer_Driven': [KilometersDriven],
    'Transmission': [Transmission],
    'Owner_No.': [OwnerNo],
    'Model': [selected_Model],
    'Model_year': [ModelYear],
    'City': [City],
    'Mileage': [Mileage],
    'Max_power': [Max_power],
    'Engine_displacement': [Engine_displacement],
    'Car_Age': [2024 - ModelYear],
    'Mileage_normalized': [Mileage / np.where((2024 - ModelYear) == 0, 1, (2024 - ModelYear))]
}

user_df = pd.DataFrame(user_input_data)

# Ensure column order matches trained model
user_df = user_df[features]

# Preprocess user input
user_df = preprocess_input(user_df)

# Apply Label Encoding with Exception Handling
for column in ['Fuel_type', 'Body_type', 'Transmission', 'Model', 'City']:
    if column in label_encoders:
        try:
            if user_df[column][0] in label_encoders[column].classes_:
                user_df[column] = label_encoders[column].transform(user_df[column])
            else:
                user_df[column] = -1  # Assign unknown values a placeholder encoding
        except Exception as e:
            st.error(f"Encoding Error: {e}")

# Prediction Button
if st.sidebar.button('Predict'):
    with st.spinner("Predicting... 🚀"):
        if user_df.notnull().all().all():
            try:
                # Make Prediction
                predicted_price = model.predict(user_df)

                # Ensure correct shape for inverse transformation
                predicted_price_norm = scalers.inverse_transform(np.array(predicted_price).reshape(-1, 1))[0][0]

                # Display Predicted Price
                st.markdown(f"""
                    <div class="result-container">
                        <h2 class="prediction-title">Predicted Car Price</h2>
                        <p class="prediction-value">₹{predicted_price_norm:,.2f}</p>
                        <p class="info">Car Age: {user_df['Car_Age'][0]} years</p>
                        <p class="info">Efficiency Score: {user_df['Mileage_normalized'][0]:,.2f} km/year</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
            st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")
