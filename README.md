# 🎮 Video Game Sales Prediction (Regression Project)

## 📘 Project Overview
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

✅ *All work is original and completed for educational purposes.*
