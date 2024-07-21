# CRV-Predictor
Welcome to the Car Resale Value Prediction Model repository! This project leverages machine learning to predict the resale value of cars based on various features. The model is deployed using Streamlit and can be accessed through the provided link.
## Dataset
I have taken the dataset from kaggle. The dataset contains detailed information about various cars, including the features. It has been preprocessed to handle missing values and encode categorical variables. EDA has also been performed. 
## Models Used
I have used Linear regression, Random Forest and XGBoost. I have chosen these models as linear regression is a simple and interpretable model used as a baseline while random forest and xgboost provide a more complex ensemble learning method that typically provides higher accuracy. There accuracies are provided below:
- Linear Regression : 70%
- XGBoost : 91%
- Random Forest : 93%
## Fututre Work
While the current model performs well with the provided dataset, it can be improved in various aspects.
Future enhancements could include some important features like model of car(basemodel or topmodel), Number of times the car has been reapaired etc. which also plays a crucial role in deciding the value of car. We could include more cars that are newly released. 

I have deployed this model using streamlit. Here is the [link https://crv-predictor-i7rinmywmpaazapp2q6bf9m.streamlit.app/]
