# Loan Approval Prediction

## 📌 Overview
This project develops and evaluates machine learning models to predict loan approval decisions based on applicant characteristics such as income, employment length, home ownership, loan purpose, and credit history.  
The goal is to maximize predictive performance while ensuring fairness, interpretability, and robustness to class imbalance.

## 🎯 Problem Statement
Loan approval is a crucial task for financial institutions, aiming to minimize defaults while not missing out on creditworthy applicants.  
The challenge lies in:
- Handling **imbalanced datasets** where most applications are denied.
- Processing **mixed data types** (numerical, categorical, binary).
- Balancing **precision and recall** for high-risk and low-risk borrowers.
- Ensuring **model interpretability** to avoid bias in decision-making.

## 📂 Dataset
- **Source**: [Kaggle – Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10/data)  
- **Rows**: 58,645  
- **Columns**: 13 (including demographic, financial, and credit history features)  
- **Target Variable**:  
  - `loan_status = 1` → Loan Approved  
  - `loan_status = 0` → Loan Denied

Key features:
- `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`
- `cb_person_default_on_file`, `cb_person_cred_hist_length`

## 🛠 Methodology
### 1. **Data Preprocessing**
- Outlier detection and removal (e.g., employment length > 100 years)
- Standardization of numerical features
- Label encoding for categorical variables
- **SMOTE** for class balancing
- **PCA** for dimensionality reduction (retained 95% variance)

### 2. **Models Used**
- **Logistic Regression** – Baseline model
- **Random Forest** – Best overall ROC-AUC performance
- **XGBoost** – Highest recall for minority class
- **Neural Network** – Strong minority recall with deep learning

### 3. **Evaluation Metrics**
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

## 📊 Results
| Model                | Accuracy | Recall (Class 1) | ROC-AUC |
|----------------------|----------|------------------|---------|
| Logistic Regression  | 0.90     | 0.42             | 0.87    |
| Random Forest        | 0.89     | 0.76             | 0.90    |
| XGBoost              | 0.84     | 0.81             | 0.89    |
| Neural Network       | 0.73     | 0.76             | 0.91    |

- **Best Overall**: Random Forest (balanced metrics)
- **Best Recall (Minority Class)**: XGBoost & Neural Network

## 💡 Key Insights
- Class imbalance was a major challenge; SMOTE significantly improved minority class recall.
- Random Forest provided the most balanced performance, making it suitable for production use.
- Feature importance analysis and SHAP values helped explain model predictions and ensure transparency.

