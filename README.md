# CRV-Predictor
Welcome to the Car Resale Value Prediction Model repository! This project leverages machine learning to predict the resale value of cars based on various features. The model is deployed using Streamlit and can be accessed through the provided link.
## Dataset
I have taken the dataset from kaggle. The dataset contains detailed information about various cars, including the features mentioned above. It has been preprocessed to handle missing values, normalize numerical features, and encode categorical variables.
## Models Used
I have used Linear regression, Random Forest and XGBoost. I have chosen these models only as linear regression is a simple and interpretable model used as a baseline while random forest and xgboost provide a more complex ensemble learning method that typically provides higher accuracy. There accuracies are provided below:
- Linear Regression : 70%
- XGBoost : 91%
- Random Forest : 93%
## Fututre Work
The model isn't perfect but is still pretty good with the provided dataset.
The projet can be improved in no. of ways.For Example The Dataset do not provide some important features like model of car(basemodel or topmodel), Number of times the car has been reapaired etc. which also plays a crucial role in deciding the value of car. We could include more cars that are newly released.

I have deployed this model using streamlit. Here is the [link https://crv-predictor-i7rinmywmpaazapp2q6bf9m.streamlit.app/]
