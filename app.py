import pandas as pd
import streamlit as st
from preprocessing import preprocess_data
from models import get_model
from visualization import (plot_prevalence_distribution,plot_records_by_year
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,    classification_report,RocCurveDisplay)
st.set_page_config(
    page_title="US Chronic Disease Analysis Project",
    layout="wide"
)
@st.cache_data
def load_processed_data():
    return preprocess_data()
df_raw, df_clean, encoders = load_processed_data()
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Preprocessing",
        "EDA",
        "ML Models",
        "Model Comparison"
    ]
)
# OVERVIEW PAGE

if page == "Overview":
    st.title("🏥 Diabetes Prevalence Prediction")

    st.markdown("""
    ### 📌 Project Overview
    This project analyzes diabetes prevalence across U.S. states using CDC public health data. It uses machine learning to classify diabetes prevalence into High and Low categories and provides interactive visualizations for healthcare analysis.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Raw Records", f"{df_raw.shape[0]:,}")
    col2.metric("Diabetes Records", f"{df_clean.shape[0]:,}")
    col3.metric("Features Used", len(df_clean.columns) - 1)
    col4.metric("ML Models", "4")

   

    summary = pd.DataFrame({
    "Feature": df_clean.columns,
    "Data Type": df_clean.dtypes.astype(str)
})
    st.write(f"**Dataset Shape:** {df_clean.shape}")
    st.dataframe(summary)

# PREPROCESSING PAGE (IMPORTANT)

elif page == "Preprocessing":
    st.title("🧹 Data Preprocessing")

    st.markdown("""
    ### Steps Performed
    1. Selected required columns  
    2. Removed missing values  
    3. Filtered one disease (Diabetes)  
    4. Created target variable (High / Low)  
    5. Encoded categorical data  
    """)

    st.subheader("Final Preprocessed Dataset")
    st.dataframe(df_clean.head())

# EDA PAGE

elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    plot_prevalence_distribution(df_clean)
    plot_records_by_year(df_clean)
# ML MODELS PAGE
elif page == "ML Models":
    st.title("🤖 Machine Learning Models")

  
    X = df_clean.drop(columns=["Prevalence_Level"])
    y = df_clean["Prevalence_Level"]

  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model_name = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest","XGBoost"]
    )

    if st.button("Train Model"):

        model = get_model(model_name, X_train, y_train)
        st.success(f"{model_name} trained successfully!")
        y_pred = model.predict(X_test)

   
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

        st.subheader("Classification Report")

        report = classification_report(
           y_test,
           y_pred,
           output_dict=True
       )

        st.dataframe(
        pd.DataFrame(report).transpose()
        )

        st.subheader("ROC Curve")

        fig_roc, ax_roc = plt.subplots()
        
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax_roc
        )
        
        st.pyplot(fig_roc)

        if model_name == "Random Forest":
            st.subheader("Feature Importance")

            feature_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(feature_df.set_index("Feature"))


# MODEL COMPARISON PAGE
elif page == "Model Comparison":
    st.title("📈 Model Comparison")

    X = df_clean.drop(columns=["Prevalence_Level"])
    y = df_clean["Prevalence_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]

    results = {}

    for name in model_names:

      model = get_model(name, X_train, y_train)

      y_pred = model.predict(X_test)

      results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    comparison_df = pd.DataFrame(results).T

    st.subheader("Performance Comparison")

    st.dataframe(
    comparison_df.style.format("{:.2f}")
)  
    best_model = comparison_df["Accuracy"].idxmax()

    st.success(f"Best Performing Model: {best_model}")
    st.subheader("Accuracy Comparison")
    st.bar_chart(comparison_df[["Accuracy"]])