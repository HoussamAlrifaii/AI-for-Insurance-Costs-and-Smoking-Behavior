![image](https://github.com/user-attachments/assets/1cf086e0-5545-4f2a-9b6c-d0a8208558a1)

# AI for Insurance: Costs and Smoking Behavior

## Introduction
This project focuses on leveraging machine learning models to predict medical insurance costs and classify individuals as smokers or non-smokers. By analyzing features such as age, BMI, and smoker status, the project provides actionable insights for healthcare cost management and risk assessment.

## Project Overview
The Insurance Cost Prediction and Smoker Classification project utilizes a structured dataset of insurance data and applies both regression and classification techniques. The primary goals are to:
1. Predict medical costs using regression models.
2. Classify individuals as smokers or non-smokers using classification models.
3. Identify the most influential factors affecting medical costs and smoking status.

---

## Dataset
- **Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance)

**Attributes**:
- **age**: Age of the primary beneficiary.
- **sex**: Insurance contractor gender (female, male).
- **bmi**: Body mass index, an indicator of body weight relative to height.
- **children**: Number of children/dependents covered by the insurance.
- **smoker**: Smoking status (yes/no).
- **region**: Residential area in the US (northeast, southeast, southwest, northwest).
- **charges**: Medical costs billed by the insurance company.

**Size**: 1,338 rows and 7 columns.

**BMI Categories**:
- 705 obese  
- 385 overweight  
- 221 healthy  
- 24 underweight  

---

## Machine Learning Models
### Regression Models
- **Linear Regression**: Predicts continuous outputs by finding the line of best fit.
- **Ridge Regression**: A regularized version of linear regression to reduce overfitting.
- **Decision Trees**: A non-linear model that splits data into subsets to minimize error.

### Classification Models
- **Logistic Regression**: Predicts binary outcomes using probabilities.
- **Decision Trees**: Intuitively classify individuals by splitting data based on feature thresholds.
- **Random Forest**: Combines multiple decision trees for robust classification.

---

## Features and Tools
- **Data Preprocessing**:
  - Label Encoding for categorical variables like `sex` and `smoker`.
  - Descriptive analysis to explore data distributions and detect missing values.
- **Visualization**:
  - Used Seaborn and Matplotlib to create histograms, scatter plots, and heatmaps.
- **Feature Importance**:
  - Identified `smoker`, `bmi`, and `age` as key predictors for both regression and classification tasks.

---

## Results
### Regression Models
| **Model**             | **MAE** | **MSE** | **RMSE** | **R² Score** |
|-----------------------|---------|---------|----------|--------------|
| **Linear Regression** | 0.066   | 0.008   | 0.092    | 0.804        |
| **Ridge Regression**  | 0.066   | 0.008   | 0.092    | 0.859        |
| **Decision Tree**     | 0.045   | 0.006   | 0.078    | 0.859        |

- Decision Tree provided the best performance with the lowest RMSE and highest R² score.

### Classification Models
| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-------------------------|--------------|---------------|------------|--------------|
| **Logistic Regression** | 89.89%       | 0.85          | 0.88       | 0.86         |
| **Decision Tree**       | 97.00%       | 0.96          | 0.97       | 0.96         |
| **Random Forest**       | 97.00%       | 0.96          | 0.97       | 0.96         |

- Random Forest slightly outperformed other models in classification accuracy.

---

## Recommendations
1. **For Cost Prediction**: Use Decision Tree Regression for the most accurate predictions.
2. **For Smoker Classification**: Random Forest provides robust and highly accurate results.

---
## Future Work
- **Enhancements**:
  - Include additional features, such as medical history, to improve prediction accuracy.
  - Explore advanced machine learning techniques like Gradient Boosting or Neural Networks.
- **Deployment**:
  - Deploy as a web application for real-time predictions and classifications.

---

## Conclusion
This project demonstrates the feasibility of using machine learning to predict medical costs and classify smokers. Key factors such as smoking status, BMI, and age were identified as significant predictors. The study highlights the utility of Decision Trees for regression tasks and Random Forests for classification tasks, making them effective tools for healthcare cost management and risk assessment.

## Installation
To run this project, install the following dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn


