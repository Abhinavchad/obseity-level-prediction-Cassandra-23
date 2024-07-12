# Obesity Risk Prediction

## **Project Overview**

Obesity has become a global health crisis, significantly affecting individuals' well-being and healthcare systems worldwide. Addressing this issue involves identifying individuals at risk and understanding the factors contributing to obesity. This project aims to predict whether a person is at risk of obesity based on various health and lifestyle factors using machine learning techniques.

For more details, visit the [Kaggle Competition](https://www.kaggle.com/competitions/cassandra24-ps-2).

## **Project Objectives**

- **Develop a Predictive Model**: Create a machine learning model to predict the risk of obesity.
- **Data Preprocessing**: Handle missing values, perform feature engineering, and normalize the dataset for effective model training.
- **Model Evaluation**: Evaluate the model's performance using metrics like the F1 score to ensure its accuracy and reliability.
- **Implementation of Best Practices**: Apply industry-standard practices for classification tasks to achieve optimal model performance.

## **Methodology**

### **1. Data Collection and Preprocessing**

Collected data from various health indicators and performed the following preprocessing steps:

- **Handling Missing Values**: Imputed missing data to ensure the completeness of the dataset.
- **Feature Engineering**: Created new features and transformed existing ones to improve model performance.
- **Normalization**: Applied one-hot encoding to convert categorical variables into a numerical format suitable for machine learning algorithms.

### **2. Model Development**

- **Model Selection**: Chose the `XGBClassifier` from the XGBoost library for its effectiveness in classification tasks.
- **Hyperparameter Tuning**: Used `GridSearchCV` to find the best hyperparameters for the model.
  ```python
  from sklearn.model_selection import GridSearchCV
  import xgboost as xgb
  
  # Define parameter grid for GridSearchCV
  param_grid = {
      'max_depth': [3],
      'learning_rate': [0.1],
      'n_estimators': [300],
  }

  # Create XGBClassifier model
  model = xgb.XGBClassifier(objective='multi:softmax', num_class=7, random_state=42)

  # Initialize GridSearchCV
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro')

  # Perform Grid Search
  grid_search.fit(one_hot_encoded_train[features], one_hot_encoded_train[target])

  # Best parameters found during grid search
  print("Best Parameters:", grid_search.best_params_)

  # Get the best model
  best_model = grid_search.best_estimator_

  # Make predictions on the test data
  test_predictions = best_model.predict(one_hot_encoded_test[features])
