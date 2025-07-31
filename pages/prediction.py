# prediction/predict.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, df):
        # Load model and scaler
        self.df = df.copy()
        self.expected_cols = pd.get_dummies(self.df.drop("HeartDisease", axis=1)).columns.tolist()
        self.lr_model = joblib.load("model/logistic_regression.pkl")
        self.xgb_model = joblib.load("model/xgboost.pkl")
        self.rf_model = joblib.load("model/random_forest.pkl")
        self.scaler = joblib.load("model/scaler.pkl")
        # self.expected_cols = joblib.load("model/feature_columns.pkl")

    def show_form(self):
        st.subheader("🩺 Enter Patient Information")

        # Replace these with your actual feature names
        age = st.number_input("Age", 20, 100, 45)
        resting_bp = st.number_input("Resting Blood Pressure", 50, 200, 120)
        cholesterol = st.number_input("Cholesterol", 50, 300, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        rest_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
        exercise_angina = st.selectbox("Exercise-induced Angina", ["Y", "N"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        if st.button("🔍 Predict Risk"):
            self.predict({
                "Age": age,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "MaxHR": max_hr,
                "Oldpeak": oldpeak,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingECG": rest_ecg,
                "ExerciseAngina": exercise_angina,
                "ST_Slope": st_slope
            })

    def predict(self, input_dict):

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode to match training features
        input_df = pd.get_dummies(input_df)


        # Ensure all expected columns exist
        # Hardcoded expected columns from training
        expected_cols = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
            'ST_Slope_Flat', 'ST_Slope_Up'
        ]
        # expected_cols = self.expected_cols
        for col in expected_cols:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[expected_cols]

        # Scale
        scaled = self.scaler.transform(input_df)

        # Make predictions for each model
        lr_pred = self.lr_model.predict(scaled)[0]
        lr_proba = self.lr_model.predict_proba(scaled)[0][1]

        xgb_pred = self.xgb_model.predict(scaled)[0]
        xgb_proba = self.xgb_model.predict_proba(scaled)[0][1]

        rf_pred = self.rf_model.predict(scaled)[0]
        rf_proba = self.rf_model.predict_proba(scaled)[0][1]


        # Display
        st.subheader("🔍 Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Logistic Regression**")
            st.write(f"Prediction: {'High Risk' if lr_pred else 'Low Risk'}")
            st.write(f"Probability: {lr_proba:.2f}")

        with col2:
            st.markdown("**Random Forest**")
            st.write(f"Prediction: {'High Risk' if rf_pred else 'Low Risk'}")
            st.write(f"Probability: {rf_proba:.2f}")

        with col3:
            st.markdown("**XGBoost**")
            st.write(f"Prediction: {'High Risk' if xgb_pred else 'Low Risk'}")
            st.write(f"Probability: {xgb_proba:.2f}")

        # Summary
        st.markdown("### Summary of Predictions")
        # 🔍 Prediction Summary
        st.markdown("### 📊 Prediction Summary (with Confidence Scores)")

        st.markdown(f"""
        - 🧮 **Logistic Regression**: {'🚨 High Risk' if lr_pred else '✅ Low Risk'}  
        &nbsp;&nbsp;&nbsp;&nbsp;➡️ **Confidence:** {lr_proba:.2%}

        - 🌲 **Random Forest**: {'🚨 High Risk' if rf_pred else '✅ Low Risk'}  
        &nbsp;&nbsp;&nbsp;&nbsp;➡️ **Confidence:** {rf_proba:.2%}

        - ⚡ **XGBoost**: {'🚨 High Risk' if xgb_pred else '✅ Low Risk'}  
        &nbsp;&nbsp;&nbsp;&nbsp;➡️ **Confidence:** {xgb_proba:.2%}
        """)

        st.markdown("### 📊 Model Evaluation Summary")

        st.markdown("**Logistic Regression**  "
        "- Accuracy: 88.6%  "
        "- Precision: 87.2%  "
        "- Recall: 93.1%  "
        "- ROC-AUC: 88.0%")

        st.markdown("**Random Forest**"
        "- Accuracy: 87.5%  "
        "- Precision: 88.3%  "
        "- Recall: 89.2%  "
        "- ROC-AUC: 87.3%")

        st.markdown("**XGBoost**  "
        "- Accuracy: 85.9%  "
        "- Precision: 87.3%  "
        "- Recall: 87.3%  "
        "- ROC-AUC: 85.7%")

        st.markdown("---")
        st.markdown("### 🧮 Confusion Matrix Snapshots")
        st.image("plots/Logistic Regression Confusion Matrix.png", caption="Logistic Regression")
        st.image("plots/Random Forest Confusion Matrix.png", caption="Random Forest")
        st.image("plots/XGBoost Confusion Matrix.png", caption="XGBoost")


def run():
    predictor = Predictor()
    predictor.show_form()
