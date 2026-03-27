# Bank-Marketing-dataset
This project aims to build a machine learning model that predicts whether a bank customer will subscribe to a term deposit based on their demographic details, financial status, and previous marketing interactions.
The dataset comes from a real-world bank marketing campaign, where customers were contacted via phone calls. The goal is to help the bank identify potential customers more efficiently, reducing unnecessary calls and improving marketing success rates. 

Key aspects of the problem:
It is a binary classification problem:
Yes (1): Client subscribes to term deposit
No (0): Client does not subscribe
The dataset is highly imbalanced (far more "no" than "yes"), making evaluation more challenging
Business objective:
Improve conversion rate
Reduce marketing cost
Enable data-driven targeting



# 📊 Bank Marketing Prediction – End-to-End ML Pipeline

## 🔍 Project Overview
This project aims to build a machine learning model that predicts whether a bank customer will subscribe to a term deposit based on their demographic details, financial information, and previous marketing interactions.

The dataset is derived from real-world bank marketing campaigns conducted via phone calls. The goal is to help the bank identify potential customers more efficiently, reduce unnecessary outreach, and improve overall campaign effectiveness.

This is a binary classification problem:
- 1 → Client subscribes to term deposit
- 0 → Client does not subscribe

The dataset is highly imbalanced, making F1-score a more reliable evaluation metric than accuracy.

---

## ⚙️ Key Contributions

### 1. Data Preprocessing & Feature Engineering
- Created a new feature `previous_contact` from `pdays` to capture prior contact information
- Dropped redundant feature `pdays`
- Separated features into:
  - Numerical features
  - Categorical features
- Built a preprocessing pipeline using ColumnTransformer:
  - Numerical pipeline:
    - Missing value imputation (mean)
    - Power transformation (to reduce skewness)
    - Standard scaling
  - Categorical pipeline:
    - Missing value imputation (most frequent)
    - One-hot encoding

---

### 2. Handling Class Imbalance
- Used SMOTE (Synthetic Minority Oversampling Technique)
- Applied SMOTE inside the pipeline to avoid data leakage
- Ensured proper resampling only on training data

---

### 3. Model Building
Implemented multiple models using pipelines:
- Logistic Regression
- Random Forest
- XGBoost

Each pipeline included:
- Preprocessing
- SMOTE
- Model training

---

## 🤖 Model Performance

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 0.8649   | 0.6018   |
| Random Forest      | 0.9105   | 0.6103   |
| XGBoost            | 0.9132   | 0.6074   |

Insights:
- Tree-based models performed better in accuracy
- F1 scores were similar due to class imbalance

---

## 🔧 Hyperparameter Tuning & Validation

- Performed 5-fold Cross-Validation:
  - Cross-Validation F1 Score: 0.5723

- Applied GridSearchCV:
  Best Parameters:
  {
    'model__max_depth': 10,
    'model__n_estimators': 100
  }

---

## 📈 Final Model Performance

- Accuracy: 0.8784
- F1 Score: 0.6202

Confusion Matrix:
[[6418  892]
 [ 110  818]]

 Here majorly tried to reduce the false Negatives

Interpretation:
- Good performance on majority class
- Improved minority class prediction using SMOTE
- Balanced precision and recall using F1-score

---

## 🔝 Feature Importance

Top Features:
- Feature 1: 0.3792
- Feature 4: 0.1213
- Feature 6: 0.0524
- Feature 5: 0.0466
- Feature 7: 0.0376
- Feature 33: 0.0339
- Feature 52: 0.0306
- Feature 41: 0.0266
- Feature 34: 0.0250
- Feature 28: 0.0187

Insights:
- Few features dominate model decisions
- Tree-based models provide interpretability

---

## 🧠 Key Learnings
- Pipelines prevent data leakage and improve reproducibility
- SMOTE must be applied inside pipeline
- Accuracy is misleading for imbalanced data
- F1-score is a better metric for evaluation
- Hyperparameter tuning improves performance
- Feature importance enhances interpretability

---

## Imp Description

Machine Learning Project – Bank Marketing Prediction
- Built an end-to-end ML pipeline using Scikit-learn and Imbalanced-learn
- Applied SMOTE to handle class imbalance within pipeline
- Implemented preprocessing using ColumnTransformer
- Trained and compared Logistic Regression, Random Forest, and XGBoost
- Performed hyperparameter tuning using GridSearchCV
- Achieved 87.8% accuracy and 0.62 F1-score (Majorly focused to reduce the false negatives)
- Analyzed feature importance for model interpretability
