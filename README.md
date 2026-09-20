# Linear Regression From Scratch

A beginner machine learning project implementing Linear Regression from scratch using NumPy, without using sklearn.linear_model.LinearRegression.
The model is trained on the California Housing dataset to predict median house values.

## Project Overview

The goal of this project is to understand how Linear Regression works internally by implementing the key components manually rather than relying on a pre-built regression model.

## The project covers:

Linear Regression
Gradient Descent
Feature Scaling
Train/Test Split
Model Prediction
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score
Actual vs. Predicted Visualization

## Workflow
California Housing Dataset
          ↓
    Train/Test Split
          ↓
     Feature Scaling
          ↓
   Linear Regression
          ↓
      Prediction
          ↓
   Model Evaluation
          ↓
 Actual vs. Predicted
    Visualization

**Dataset**

This project uses the California Housing dataset provided through scikit-learn.
The dataset contains information about California housing districts, including demographic and housing-related features, with the goal of predicting median house values.

**Model Implementation**

The Linear Regression model is implemented from scratch using NumPy.
The training process uses Gradient Descent to iteratively update the model's weights and bias in order to minimize the prediction error.
Feature scaling is also implemented to help Gradient Descent converge more effectively.

# Model Evaluation

The model is evaluated using three common regression metrics:

## Mean Squared Error (MSE)**
Measures the average squared difference between actual and predicted values.

## Mean Absolute Error (MAE)**
Measures the average absolute difference between actual and predicted values.

## R² Score
Measures how well the model explains the variance in the target variable.

## Visualization
The project includes an Actual vs. Predicted plot to visualize model performance.
The red reference line represents perfect predictions:
Actual = Predicted
Points closer to this line indicate that the predictions are closer to the actual values.

# Technologies
Python
NumPy
Pandas
Matplotlib
Scikit-learn
scikit-learn is used for the dataset and supporting utilities, but LinearRegression from sklearn.linear_model is not used.

**Project Structure**
.
├── linear_regression.py
├── california_housing.py
├── requirements.txt
└── README.md

**Files**
linear_regression.py — Contains the Linear Regression and StandardScaler implementations.
california_housing.py — Loads the California Housing dataset, trains the model, evaluates predictions, and generates visualizations.
requirements.txt — Lists the required Python dependencies.
README.md — Project documentation.

**What I Learned**
This project helped me understand how Linear Regression works behind the scenes.
Through this implementation, I learned:
How Linear Regression makes predictions using weights and bias
How Gradient Descent updates model parameters
Why feature scaling is important for optimization
How to split data into training and testing sets
How MSE, MAE, and R² are used to evaluate regression models
How to visualize actual versus predicted values

Next Step
The next step in my machine learning journey is implementing Logistic Regression from scratch and continuing to build core machine learning algorithms without relying entirely on pre-built models.
