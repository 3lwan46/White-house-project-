# 🎮 Video Game Sales Prediction (Regression Part)

## 📘 Overview
This project applies **Supervised Machine Learning** techniques to predict **global video game sales** based on categorical features such as platform, genre, and publisher.  
The goal is to build simple and explainable (white-box) models using **Linear Regression** and **K-Nearest Neighbors (KNN)**.

---

## 🧩 Dataset
- **Source:** VGChartz.com (games with >100k sales)
- **Rows:** ~16,000  
- **Columns:**
  - Platform, Year, Genre, Publisher
  - Regional Sales (NA, EU, JP, Other)
  - Global_Sales (target)

**Target Variable:** `Global_Sales`  
**Features Used:** `Platform`, `Genre`, `Publisher`  
(Excluded `Year`, `Rank`, and regional sales to avoid data leakage.)

---

## ⚙️ Workflow Summary
### 1️⃣ Data Preparation
- Checked for missing values and outliers.  
- Dropped non-useful columns (`Rank`, regional sales).  
- Handled missing values with `SimpleImputer`.  
- Encoded categorical features using **OneHotEncoder**.  
- Scaled data for KNN using **StandardScaler**.

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized distributions, outliers, and trends.  
- Found strong correlations between regional and global sales.  
- Noted that most games have low sales; few have very high sales.

### 3️⃣ Model Building
**Model 1 – Linear Regression (V1):**  
- Baseline using `Platform`, `Genre`, `Publisher`.  
- R² ≈ 0.085, RMSE ≈ 1.98  

**Model 2 – KNN Regressor:**  
- **V1:** Default (k=5) → R² ≈ 0.05, RMSE ≈ 2.01  
- **V2:** Tuned (k=11, uniform weights) → **R² ≈ 0.106**, **RMSE ≈ 1.96**

### 4️⃣ Model Evaluation
- Metrics: **R² Score**, **RMSE**, and **Residual Plots**.  
- KNN (V2) achieved slightly better numerical results.  
- Linear Regression remains easier to interpret (white-box).

---

## 🏁 Final Model & Insights
- **Best Performing Model:** KNN Regressor (V2, tuned)  
- **Most Explainable Model:** Linear Regression (V1)  
- Models show moderate generalization (low R² values).  
- Data lacks strong predictors — future versions should include numeric or behavioral features (e.g., ratings, marketing spend).

---

## 🧠 Key Learnings
- Encoding and scaling are essential for categorical data and KNN models.  
- Even simple models reveal useful trends about platforms and genres.  
- Model tuning (GridSearchCV) improves performance modestly.  
- “White-box” models like Linear Regression are ideal for non-technical stakeholders.

---

## 🧰 Tools & Libraries
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (LinearRegression, KNeighborsRegressor, GridSearchCV)

---


---

Classification part: Credit Score Prediction
🎯 Problem Statement

A global financial institution wants to automate its credit scoring process to reduce human bias and improve accuracy.
The goal is to classify customers into three categories — Poor, Standard, Good — based on financial and behavioral attributes.

🧩 Dataset Overview

Dataset: train.csv from Kaggle’s Credit Score Classification dataset

Total Rows: ~53,000

Features: 28 (demographic, income, loan, and payment history)

Target: Credit_Score (3 classes: Poor, Standard, Good)

⚙️ Data Preparation

Removed irrelevant columns: ID, Customer_ID, Name, SSN, Month

Handled missing values:

Numeric → filled with mean

Categorical → filled with most frequent value

Encoded categorical features using LabelEncoder

Scaled numeric features with StandardScaler (important for KNN)

Train-Test Split: 80% training, 20% testing (stratified)

🧪 Models Implemented
1️⃣ Logistic Regression

Version 1: Baseline model

Version 2: Tuned with GridSearchCV (C, solver)

Result:

Accuracy ≈ 0.58

F1 Score ≈ 0.50

Interpretation: Performs poorly due to complex non-linear relationships in data, but remains useful for explainability.

2️⃣ K-Nearest Neighbors (KNN)

Version 1: Baseline with k=5

Version 2: Tuned with GridSearchCV

Best Params → n_neighbors=3, weights='distance', metric='manhattan'

Result:

Accuracy ≈ 0.73

F1 Score ≈ 0.72

Interpretation: Captures non-linear relationships effectively and generalizes well.

📊 Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression (v1)	0.58	0.56	0.49	0.50
Logistic Regression (v2)	0.58	0.56	0.49	0.50
KNN (v1)	0.60	0.57	0.58	0.57
KNN (v2 - Tuned)	0.73	0.72	0.71	0.72
🧠 Insights

Best Model: KNN (Version 2 - Tuned)

Reason: Achieved the best balance between accuracy, recall, and precision.

Feature scaling and tuning had a major impact on KNN performance.

No overfitting observed — consistent training and testing performance.

Logistic Regression remains valuable as a white-box model for transparency and interpretability.

🏁 Final Notes

The project demonstrates the full supervised learning workflow — from data cleaning and feature engineering to model evaluation and tuning.
For classification tasks with non-linear relationships, KNN proved more effective than Logistic Regression.
Future work may include:

Trying tree-based models (e.g., Random Forest, XGBoost)

Applying feature importance analysis

Testing imbalanced class handling (e.g., SMOTE)
