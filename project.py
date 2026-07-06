import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="US Chronic Disease Analysis Project",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\asus\Downloads\U.S._Chronic_Disease_Indicators (1).csv")

df = load_data()

df_raw = df.copy()

# DATA PREPROCESSING (GLOBAL)

df_clean = df[[
    "YearStart",
    "LocationDesc",
    "Topic",
    "DataValue"
]].copy()

df_clean.dropna(inplace=True)

df_clean = df_clean[df_clean["Topic"] == "Diabetes"]

threshold = df_clean["DataValue"].median()

df_clean["Prevalence_Level"] = np.where(
    df_clean["DataValue"] >= threshold, 1, 0
)

df_clean.drop(columns=["Topic", "DataValue"], inplace=True)

le = LabelEncoder()
df_clean["LocationDesc"] = le.fit_transform(df_clean["LocationDesc"])

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Preprocessing", "EDA", "ML Models", "Model Comparison"]
)

# OVERVIEW PAGE

if page == "Overview":
    st.title("🏥 Diabetes Prevalence Prediction")

    st.markdown("""
    ### 📌 Project Overview
    This project predicts whether **diabetes prevalence**
    in U.S. states is **High or Low** using machine learning.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Rows", df_raw.shape[0])
    col2.metric("Clean Rows", df_clean.shape[0])
    col3.metric("Target", "Prevalence_Level")

    st.subheader("📄 Raw Data Sample")
    st.dataframe(df_raw.head())

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

    st.subheader("Prevalence Distribution")
    fig, ax = plt.subplots()
    df_clean["Prevalence_Level"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Low", "High"], rotation=0)
    st.pyplot(fig)

    st.subheader("Records by Year")
    fig2, ax2 = plt.subplots()
    df_clean.groupby("YearStart").size().plot(ax=ax2)
    st.pyplot(fig2)

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
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if st.button("Train Model"):

     
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = RandomForestClassifier(random_state=42)

        
        model.fit(X_train, y_train)

     
        y_pred = model.predict(X_test)

   
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {acc:.2f}")

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=ax_cm
        )
        st.pyplot(fig_cm)

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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    st.table(pd.DataFrame.from_dict(
        results, orient="index", columns=["Accuracy"]
    ))
