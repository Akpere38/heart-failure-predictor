import streamlit as st
import pandas as pd
from pages.insights import InsightsPage
from pages.prediction import Predictor


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["Insights", "Predict"], index=0)



if page == "Insights":
    InsightsPage(df).render()

elif page == "Predict":
    Predictor(df).show_form()

