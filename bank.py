import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import shap

# Load the dataset (use a Kaggle banking dataset)
df = pd.read_csv("bank_marketing.csv")

# Data exploration
def explore_data(df):
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df['y'].value_counts(normalize=True))
    
    # Visualize target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='y', data=df)
    plt.title('Target Distribution')
    plt.savefig('target_distribution.png')
    
    # Analyze age distribution by target
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='y', element='step', common_norm=False)
    plt.title('Age Distribution by Subscription Status')
    plt.savefig('age_distribution.png')
    
    # Analyze job distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(y='job', hue='y', data=df, order=df['job'].value_counts().index)
    plt.title('Job Distribution by Subscription Status')
    plt.tight_layout()
    plt.savefig('job_distribution.png')
    
    return df

# Data preprocessing
def preprocess_data(df):
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y'].map({'yes': 1, 'no': 0})
    
    # Identify categorical and numerical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Define preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)
    
    return X_train, X_test, y_train, y_train_resampled, y_test, preprocessor

# Model building and evaluation
def build_evaluate_models(X_train, X_test, y_train, y_train_resampled, y_test, preprocessor):
    # Define models
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        # Train model with resampled data
        model.fit(X_train, y_train_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {results[name]["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png')
    
    # Get best model based on ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = models[best_model_name]
    
    # Feature importance analysis for the best model
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        # Extract feature names
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'cat':
                feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
        
        # Get feature importances
        importances = best_model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {best_model_name}')
        plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{best_model_name.replace(" ", "_").lower()}.png')
        
        # SHAP values for the best model (on a sample)
        sample_X = X_test.iloc[:100]  # Take a sample for visualization
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
        shap_values = explainer.shap_values(preprocessor.transform(sample_X))
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, preprocessor.transform(sample_X), feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
    
    return results, best_model_name, best_model

# Hyperparameter tuning for the best model
def tune_best_model(X_train, y_train_resampled, best_model_name, preprocessor):
    if best_model_name == 'Random Forest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    
    else:  # Logistic Regression
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
        base_model = LogisticRegression(max_iter=1000)
    
    # Create pipeline with the selected model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid, 
        n_iter=20, cv=5, scoring='roc_auc',
        random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train_resampled)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best ROC AUC: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_

# Main workflow
def main():
    df = pd.read_csv("bank_marketing.csv")
    df = explore_data(df)
    X_train, X_test, y_train, y_train_resampled, y_test, preprocessor = preprocess_data(df)
    results, best_model_name, best_model = build_evaluate_models(X_train, X_test, y_train, y_train_resampled, y_test, preprocessor)
    
    print("Model performance:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    print(f"\nBest model: {best_model_name}")
    
    # Tune the best model
    tuned_model = tune_best_model(X_train, y_train_resampled, best_model_name, preprocessor)
    
    # Final evaluation with tuned model
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    
    print("\nTuned model performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()