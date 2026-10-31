# 🏦 CreditRisk AI – Intelligent Credit Scoring

## 📌 Project Overview

**CreditRisk AI** is a Machine Learning project designed to estimate the probability of credit default using customer financial and personal information.

The main objective is to develop an intelligent credit scoring system that helps analyze credit risk through predictive modeling and explainable AI techniques.

The project uses the **Statlog German Credit Dataset** and compares several Machine Learning classification algorithms to identify the most suitable model for credit risk prediction.

## 🎯 Project Objectives

- Explore and analyze credit data using Exploratory Data Analysis (EDA).
- Perform data preprocessing and feature engineering.
- Develop and compare Machine Learning classification models.
- Predict the probability of customer credit default.
- Evaluate model performance using classification metrics.
- Interpret model predictions using Explainable AI (XAI).
- Develop an interactive application using Streamlit.

## 📊 Dataset

**Dataset:** Statlog German Credit Data

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Dataset characteristics:

- 1,000 observations.
- 20 explanatory variables.
- 1 target variable.
- Numerical and categorical features.

**Target variable:**
- `0`: Good Credit (Good customer).
- `1`: Bad Credit (Bad customer).

## ⚙️ Technologies and Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model persistence | Joblib |
| Explainable AI | Permutation Importance, Logistic Regression Coefficients |
| Application | Streamlit |
| Development environment | Google Colab |
| Version control | Git, GitHub |

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset inspection and understanding.
- Statistical analysis.
- Missing values and duplicate checks.
- Target variable distribution.
- Correlation analysis.
- Data visualization.

### 2. Data Preprocessing
- Separation of numerical and categorical variables.
- Missing value imputation.
- Numerical feature standardization.
- Categorical feature encoding using One-Hot Encoding.
- Train/Test split (80% / 20%).

### 3. Machine Learning Modeling

Three classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### 4. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### 5. Model Optimization
- Classification threshold analysis.
- Cross-validation using Stratified K-Fold.
- Model performance comparison.

### 6. Explainable AI (XAI)
- Permutation Feature Importance.
- Logistic Regression coefficient analysis.
- Individual prediction contribution analysis.

### 7. Interactive Application

A Streamlit application allows users to:

- Enter customer information.
- Estimate the probability of credit default.
- Obtain a predicted risk classification.
- Explore the prediction results.

## 📈 Model Performance

Initial results obtained on the test dataset:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.750 | 0.558 | 0.800 | 0.658 | 0.806 |
| Decision Tree | 0.665 | 0.441 | 0.433 | 0.437 | 0.599 |
| Random Forest | 0.760 | 0.675 | 0.383 | 0.489 | 0.795 |

**Selected model: Logistic Regression**

Logistic Regression was selected as the main model because it achieved the highest Recall (0.80) and ROC-AUC (0.806), making it a relevant choice when detecting potentially risky customers is a priority.

The final classification threshold was 0.50.

## 📁 Project Structure

```text
CreditRisk-AI/
│
├── app.py
├── credit_risk_model.pkl
├── metadata.json
├── requirements.txt
├── README.md
│
└── CreditRisk_AI_Training.ipynb
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CreditRisk-AI.git
```

### 2. Navigate to the project directory

```bash
cd CreditRisk-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Key Results

This project demonstrates how Machine Learning can be applied to credit risk assessment.

The comparison highlights the trade-off between detecting risky customers (Recall) and correctly identifying them (Precision).

Logistic Regression provides a useful balance of predictive performance and interpretability for this educational credit scoring application.

## ⚠️ Limitations

- The German Credit Dataset is relatively small and historical.
- The dataset does not necessarily represent current banking customers.
- Model performance depends on the selected training and test samples.
- Predictions are statistical estimates and do not establish causality.
- The application is an academic prototype and should not be used for real banking or lending decisions.

## 🔮 Future Improvements

- Hyperparameter optimization.
- More advanced classification algorithms.
- SHAP-based model explanations.
- Cost-sensitive learning.
- Advanced threshold optimization.
- Model monitoring and performance tracking.
- Integration with real-world credit data.
- Deployment and continuous improvement.

## 👩‍💻 Author

**CreditRisk AI – Intelligent Credit Scoring**

Developed as an academic Data Science project focused on Machine Learning, predictive analytics, and credit risk management.

---

⭐ If you find this project interesting, feel free to explore the repository!
