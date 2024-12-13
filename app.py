import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             mean_squared_error, r2_score, accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.svm import SVC
import shap
import joblib

# App Layout
st.title("Interactive AutoML Platform")
st.sidebar.title("AutoML Configuration")
st.markdown("This platform lets you upload datasets, preprocess them, train models, and visualize results.")

# File Upload
data_file = st.sidebar.file_uploader("Upload Your Dataset (CSV)", type=["csv"])

if data_file:
    try:
        data = pd.read_csv(data_file)
        st.write("### Uploaded Dataset", data.head())

        # Target Variable Selection
        target_column = st.sidebar.selectbox("Select the Target Variable", data.columns)

        # Error Handling for Target Column
        if data[target_column].isnull().any():
            st.error(f"Target column '{target_column}' contains missing values. Please handle them before proceeding.")

        # Customizable Preprocessing
        st.sidebar.subheader("Preprocessing Options")
        imputation_strategy = st.sidebar.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
        scaler_type = st.sidebar.selectbox("Scaling Technique", ["StandardScaler", "MinMaxScaler"])
        drop_low_variance = st.sidebar.checkbox("Drop Low Variance Features")

        # Handle Missing Values
        imputer = SimpleImputer(strategy=imputation_strategy)
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = imputer.fit_transform(data[[col]])

        # Encode Categorical Variables
        for col in data.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

        # Drop Low Variance Features
        if drop_low_variance:
            variance_threshold = 0.01
            low_variance_cols = [col for col in data.columns if data[col].var() < variance_threshold]
            data = data.drop(columns=low_variance_cols)
            st.write(f"Dropped Low Variance Features: {low_variance_cols}")

        # Scaling Numerical Features
        scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove(target_column)
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        st.write("### Preprocessed Dataset", data.head())

        # Train-Test Split
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        st.header("Model Training")
        task = st.sidebar.radio("Select Task Type", ["Classification", "Regression"])

        models = []
        if task == "Classification":
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Random Forest", RandomForestClassifier()),
                ("XGBoost", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ("SVM", SVC(probability=True))
            ]
        else:
            models = [
                ("Linear Regression", LinearRegression()),
                ("Random Forest", RandomForestRegressor()),
                ("XGBoost", xgb.XGBRegressor())
            ]

        # Hyperparameter Tuning
        st.sidebar.subheader("Hyperparameter Tuning")
        perform_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning")

        results = {}
        for name, model in models:
            if perform_tuning:
                param_grid = {
                    "Random Forest": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]},
                    "XGBoost": {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 0.2]},
                    "SVM": {"C": [0.1, 1, 10]}
                }
                if name in param_grid:
                    grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='accuracy' if task == "Classification" else 'r2')
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') if hasattr(model, 'predict_proba') else None
                results[name] = {
                    'Accuracy': acc,
                    'F1 Score': f1,
                    'ROC AUC': auc
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {
                    'MSE': mse,
                    'R2 Score': r2
                }

        # Model Comparison Chart
        st.header("Results")
        st.write("### Model Performance")
        performance_df = pd.DataFrame(results).T
        st.dataframe(performance_df)
        st.bar_chart(performance_df)

        # Output Model Interpretability
        if task == "Classification":
            for name, model in models:
                if name in results:
                    st.write(f"### {name} - Confusion Matrix")
                    cm = confusion_matrix(y_test, model.predict(X_test))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    st.pyplot(plt)

        # SHAP Explainability
        st.header("Explainability")
        if "Random Forest" in [name for name, _ in models]:
            explainer = shap.TreeExplainer(models[1][1])  # Assuming Random Forest is the second model
            shap_values = explainer.shap_values(X_test)
            st.write("### Feature Importance")
            shap.summary_plot(shap_values, X_test)
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
