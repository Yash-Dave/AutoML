# Interactive AutoML Platform

## Overview
A simple, interactive AutoML platform built with Streamlit. Upload datasets, preprocess them, train machine learning models, and visualize results for both classification and regression tasks.

## Features
- Upload datasets in CSV format.
- Preprocess data with options for imputation, scaling, and feature selection.
- Train models for classification (Logistic Regression, Random Forest, XGBoost, SVM) and regression (Linear Regression, Random Forest, XGBoost).
- Optional hyperparameter tuning.
- Visualize model performance, confusion matrices, and SHAP-based feature importance.

## Getting Started
1. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost shap joblib
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.

## Usage
1. Upload a CSV dataset.
2. Configure preprocessing options in the sidebar.
3. Train models and optionally enable hyperparameter tuning.
4. View results and visualizations directly in the app.

## Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a pull request.

## License
This project is licensed under the MIT License.

