# 🚙 Car Dekho-Used Car Price Prediction Project :

## Project overview:
This repository contains a comprehensive Machine learning project that includes historical data on used car prices from CarDekho, including various features such as mileage, model, Modelyear, fuel type, transmission type, and other relevant attributes from different cities. 
this project aims to create an accurate and user-friendly tool that enhances customer experience and streamlines the pricing process for sales representatives. The final product is a deployed Streamlit application that allows users to input car details and receive instant price predictions.
## Technical tags:
* Python
* Pandas
* Numpy
* Scikit-learn
* Plotly
* Machine learning
* Streamlit

## Structure of the project:
### 1) jupyter notebook:
File:cardekho_project3.ipynb
Description: This notebook documents the data cleaning process, feature engineering, model selection, training, and evaluation. The notebook includes exploratory data analysis (EDA) to understand the key features influencing car prices, followed by multiple machine learning models, such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, with hyperparameter tuning for optimal performance.
### 2) Streamlit Application :
File: car_dekho.py
Description: The Streamlit application provides an interactive interface where users can input car specifications and obtain a predicted price. The app is designed to be intuitive and easy to use, making it accessible for both customers and sales representatives.
### 3) Project Report :
File:cardekho_project.pdf
Description: A detailed report that covers the entire project lifecycle, from problem statement and data preprocessing to model evaluation and deployment. It includes justifications for the chosen methodologies, a summary of results, and insights derived from the analysis.
### 4) Dataset : 
Description: It contains all six city data on used car prices from CarDekho. 
## Procedures:

### 1) Data cleaning, Model Training, and Evaluation:
Data Cleaning: The dataset undergoes rigorous cleaning to handle missing values, encode categorical features, and scale numerical features appropriately. after that feature selection is done by using Random ForestModel
Model Selection: Several models were trained and evaluated, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting. Hyperparameter tuning was performed to optimize model performance.
Evaluation Metrics: The models were evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values. These metrics helped in comparing model performance and selecting the best model for deployment.

### 2) streamlit Application:
Input Fields: The application allows users to input various car attributes such as  model, model, Body type, year, fuel type, transmission type, mileage, and more.
Price Prediction: After entering the details, click the 'Predict' button to get the estimated price of the car.
User-Friendly Interface: The interface is designed to be intuitive, making it easy for both tech-savvy and non-technical users to interact with the model.

## Results:
Best Model: The model with the highest R² score and lowest MSE and MAE was selected for the final deployment.
Accuracy: The deployed model demonstrated strong predictive accuracy, making it reliable for estimating car prices.

## References :
* Python Documentation :(https://docs.python.org/3/)
* EDA Documentation :(https://python-data-science.readthedocs.io/en/latest/exploratory.html)
* Scikit-learn documentation:(https://scikit-learn.org/stable/user_guide.html)
* Streamlit Documentation:(https://docs.streamlit.io/get-started)
