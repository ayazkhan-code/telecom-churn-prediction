# 📉 Telecom Customer Churn Prediction

> Predicting high-value customer churn using Machine Learning — and identifying the key drivers behind it.

---

## 🧩 Problem Statement

Customer churn is one of the biggest revenue threats in the telecom industry. Acquiring a new customer costs 5–10× more than retaining an existing one. This project builds a machine learning pipeline to:

1. **Predict** whether a high-value customer will churn
2. **Identify** the most important features driving churn — enabling targeted business interventions

---

## 📁 Project Structure

```
telecom-churn-prediction/
│
├── churn_prediction.ipynb    # Main notebook — EDA, modeling, evaluation
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── submission.csv            # Final predictions
├── data_dictionary.csv       # Feature descriptions
├── subjective_answers.pdf    # Business Q&A and recommendations
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** Kaggle Telecom Churn Dataset
- **Size:** ~70,000 rows, 170+ features
- **Target:** `churn_probability` (binary: 0 = retained, 1 = churned)
- **Features include:** Monthly call usage (MOU), ARPU, recharge frequency, roaming minutes, night calls, and more
- **Preprocessing:** Median imputation for missing values, standard scaling, stratified train-test split (70/30)

---

## ⚙️ Models Built

### Model 1 — Logistic Regression with PCA
- Dimensionality reduced to 10 principal components
- Optimized for prediction speed and generalization

### Model 2 — Random Forest Classifier ⭐ Best Model
- Trained on full feature set (no PCA) for interpretability
- Used for **feature importance extraction** — critical for business insights

### Model 3 — Logistic Regression (No PCA)
- Baseline comparison against the PCA variant

---

## 📈 Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression + PCA | 89.8% | 51.7% | 1.4% | 0.794 |
| **Random Forest** | **94.0%** | **79.0%** | **56.4%** | **0.928** |
| Logistic Regression (No PCA) | 93.0% | 78.4% | 42.9% | 0.908 |

> **Random Forest** achieved the best overall performance with a ROC-AUC of **0.928**, making it the chosen model for churn prediction.

---

## 🔑 Top Churn Indicators (Random Forest Feature Importance)

1. Total incoming/outgoing minutes of usage (MOU)
2. Average Revenue Per User (ARPU)
3. Recharge frequency and amount
4. Roaming minutes
5. Night call volume

---

## 💡 Business Recommendations

- 🎯 **Targeted retention offers** for high-ARPU customers showing declining usage
- 📉 **Early warning system** for customers with dropping call minutes month-over-month
- 🌐 **Network quality investment** in regions with high churn concentration
- 💳 **Personalized recharge plans** for customers with irregular recharge behavior

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-purple)

- **Language:** Python 3.11
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Techniques:** PCA, Random Forest, Logistic Regression, StandardScaler, ROC-AUC evaluation

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook churn_prediction.ipynb
```

---


