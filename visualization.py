import matplotlib.pyplot as plt
import streamlit as st


def plot_prevalence_distribution(df_clean):
    st.subheader("Prevalence Distribution")

    fig, ax = plt.subplots()

    df_clean["Prevalence_Level"].value_counts().plot(
        kind="bar",
        ax=ax
    )
    ax.set_title("Distribution of Diabetes Prevalence")
    ax.set_xticklabels(["Low", "High"], rotation=0)
    ax.set_xlabel("Prevalence Level")
    ax.set_ylabel("Count")

    st.pyplot(fig)
    plt.close(fig)

def plot_records_by_year(df_clean):
    st.subheader("Records by Year")

    fig, ax = plt.subplots()

    df_clean.groupby("YearStart").size().plot(ax=ax)
    ax.set_title("Records Available by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)
    plt.close(fig)