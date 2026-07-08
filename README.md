Diabetes Prevalence Prediction
A Machine Learning project that predicts **diabetes prevalence levels** using the **CDC U.S. Chronic Disease Indicators** dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and an interactive Streamlit dashboard.

Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Interactive Streamlit dashboard
- Multiple Machine Learning models
  - Logistic Regression
  - Decision Tree
  - Random Forest (with GridSearchCV)
  - XGBoost
- Model performance comparison
- ROC Curve and Feature Importance visualization

Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit

 Dataset
Source: CDC U.S. Chronic Disease Indicators Dataset

The dataset contains diabetes prevalence information across U.S. states. After preprocessing, categorical features were encoded and the target variable was classified into **High** and **Low** prevalence levels.

Results
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve

A comparison dashboard highlights the best-performing model.
Live Demo:https://diabetes-prevalence-prediction.streamlit.app/

