# 🏦 Bank Term Deposit Subscription Prediction
A classification project using a Kaggle banking dataset to predict term deposit subscriptions. It includes data preprocessing, model training (Logistic Regression, Random Forest, Gradient Boosting), SMOTE for imbalanced classes, and model evaluation.


This project implements a complete machine learning pipeline to predict whether a customer will subscribe to a term deposit, based on the Bank Marketing dataset. It features:

- 📊 Exploratory Data Analysis (EDA)
- 🛠 Feature Preprocessing and SMOTE Oversampling
- 🤖 Multiple Classification Models
- 🧠 SHAP Explainability
- 🔍 Hyperparameter Tuning (RandomizedSearchCV)
- 📈 Performance Metrics and Visualizations

---

## 🔧 Project Structure

### 📁 Output Files
The script generates the following visualizations and evaluation reports:

- `target_distribution.png` – Subscription target distribution
- `age_distribution.png` – Age vs. subscription status
- `job_distribution.png` – Job type vs. subscription status
- `confusion_matrix_<model>.png` – Confusion matrix for each model
- `roc_curve_<model>.png` – ROC curve per model
- `feature_importance_<model>.png` – Feature importances (for tree-based models)
- `shap_summary.png` – SHAP value summary for the best model

---

## 🧠 Models Evaluated

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Each model is evaluated on the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

## 📦 Requirements

To run the project, use the included environment:

```bash
conda env create -f bank-term-deposit.yml
conda activate bank-term-deposit
