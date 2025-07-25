# 🔄 Customer Churn Prediction with XGBoost

Customer churn — when existing customers stop doing business with a company — is one of the most critical challenges across industries. Retaining a customer is far more cost-effective than acquiring a new one. That’s why **churn prediction is a high-value business task** that directly impacts revenue and customer lifetime value.

This project uses the **Telco Customer Churn** dataset and builds an optimized **XGBoost model** to predict the likelihood of customer churn. With the right preprocessing and model tuning, this system enables telecom companies (and similar businesses) to proactively retain at-risk customers.

---

## 🎯 Objectives

- Predict whether a customer will churn using their demographic and service data.
- Use XGBoost and Grid Search for hyperparameter tuning.
- Visualize important features that drive churn behavior.
- Save the model and its feature structure for real-time or batch prediction.

---

## 💡 Why Churn Prediction Matters

- 💰 **Reduce revenue loss** by identifying customers likely to leave.
- 🎯 **Target retention offers** to high-risk users.
- 📈 **Improve customer satisfaction** with proactive service.
- ⚙️ **Enable data-driven decisions** in marketing and support.

---

## 🧠 ML Pipeline Overview

1. **Data Preprocessing**
   - Convert `TotalCharges` to numeric and handle missing values
   - Label encode binary columns (`Yes/No`, `Male/Female`)
   - One-hot encode multi-class categorical features
   - Split into training and testing sets

2. **Model Training**
   - Use `XGBClassifier` from XGBoost
   - Tune `n_estimators`, `max_depth`, `learning_rate`, and `subsample` using `GridSearchCV`
   - Evaluate using ROC-AUC

3. **Model Evaluation**
   - Classification report and confusion matrix
   - ROC-AUC Score
   - Feature importance plot (Top 10)

4. **Persistence**
   - Save trained model (`xgb_model.pkl`)
   - Save training column names (`model_columns.pkl`) to match prediction input

---

## 📁 Project Structure
```
CHURN/
│
├── __pycache__/                 # Python bytecode (auto-generated, ignore in Git)
├── C/                           # Unclear folder - likely IDE-related or temp
│
├── .gitignore                   # Good: Keeps repo clean
├── churn.csv                    # Raw dataset
├── data_analysis.ipynb          # Main notebook for training & evaluation
├── model_columns.pkl            # Saved feature columns (for consistent prediction)
├── predictor.py                 # Script with model training logic
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
└── xgb_model.pkl                # Trained XGBoost model
```


---

## 🚀 How to Use This Project

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt


Churn_Prediction_Notebook.ipynb
