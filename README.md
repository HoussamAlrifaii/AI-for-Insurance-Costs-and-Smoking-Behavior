# Insurance Cost Prediction and Smoker Classification

## Introduction
This project focuses on leveraging machine learning models to predict medical insurance costs and classify individuals as smokers or non-smokers. By analyzing features such as age, BMI, and smoker status, the project aims to provide actionable insights for healthcare cost management and risk assessment.

## Project Overview
The Insurance Cost Prediction and Smoker Classification project utilizes a structured dataset of insurance data, incorporating regression and classification techniques. The primary goals are to accurately predict medical charges and classify smoker status while highlighting the most impactful features influencing these outcomes.

### Key Objectives
1. Predict medical costs using regression models.
2. Classify individuals as smokers or non-smokers using classification models.
3. Analyze feature importance to identify significant factors in predictions and classifications.

## Features
### Data Preprocessing
- **Label Encoding**: Converts categorical variables (e.g., smoker, sex) into numerical format for machine learning models.
- **Exploratory Data Analysis**: Identifies patterns and correlations among features like age, BMI, and charges.

### Regression Modeling
- Implements models such as Linear Regression, Ridge Regression, and Decision Trees to predict medical charges.
- Compares model performance using metrics like MAE, MSE, RMSE, and R² score.

### Classification Modeling
- Uses Logistic Regression, Decision Trees, and Random Forest for smoker classification.
- Evaluates model accuracy, precision, recall, and F1 score.

### Feature Importance
- Highlights the top factors influencing predictions, with smoker status, BMI, and age being the most impactful.

## Dataset
**Dataset Overview**:
- **Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/)
- **Attributes**:
  - **Numerical**: age, BMI, charges.
  - **Categorical**: sex, smoker, region.
  - **Target Variables**: charges (regression), smoker (classification).
- **Size**: 1,338 rows and 7 columns.

---

## Structure
```plaintext
Insurance-Cost-Prediction/
├── dataset/
│   └── insurance.csv
├── notebook/
│   └── MedicalCostPrediction.ipynb
├── report/
│   └── MedicalCostReport.pdf
├── README.md
