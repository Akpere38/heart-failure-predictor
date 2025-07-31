import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import shap
import numpy as np

class InsightsPage:
    def __init__(self, df):
        self.df = df

    def heart_disease_distribution(self):
        st.subheader("🫀 Heart Disease Distribution")
        heart_distri = px.histogram(
            self.df,
            x="HeartDisease",
            color="HeartDisease",
            barmode="group",
            color_discrete_map={0: 'green', 1: 'red'},
            title="Heart Disease Distribution (0 = No, 1 = Yes)"
        )
        st.plotly_chart(heart_distri, use_container_width=True)

    def age_distribution(self):
        st.subheader("📈 Age Distribution by Heart Disease Status")
        age_distrib = px.histogram(
            self.df,
            x="Age",
            color="HeartDisease",
            nbins=30,
            barmode="overlay",
            color_discrete_map={0: 'green', 1: 'red'},
            title="Age Distribution: Red = Heart Disease, Green = No Heart Disease"
        )
        age_distrib.update_traces(opacity=0.6)
        st.plotly_chart(age_distrib, use_container_width=True)

#   # Categorical Features vs Target    
    def categorical_vs_target(self):
        st.subheader("📊 Categorical Features vs Heart Disease")
        
        categorical_features = [
            "Sex", "ChestPainType", "RestingECG", 
            "ExerciseAngina", "ST_Slope"
        ]

        for feature in categorical_features:
            st.markdown(f"#### 🔹 {feature}")
            cat_vs_target = px.histogram(
                self.df,
                x=feature,
                color="HeartDisease",
                barmode="group",
                color_discrete_map={0: 'green', 1: 'red'},
                title=f"{feature} vs Heart Disease"
            )

            cat_vs_target_percent = px.histogram(
                self.df, x=feature, color="HeartDisease", barmode="relative", barnorm='percent',
                title=f"{feature} vs Heart Disease (%)")
            

            st.plotly_chart(cat_vs_target, use_container_width=True)
            st.plotly_chart(cat_vs_target_percent, use_container_width=True)


#    # Numerical Features vs Target
    def numerical_vs_target(self):
        st.subheader("📈 Numerical Features vs Heart Disease")
        
        numerical_features = [
            "Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"
        ]

        for feature in numerical_features:
            st.markdown(f"#### 🔹 {feature}")
            fig_1 = px.histogram(
                self.df,
                x=feature,
                color="HeartDisease",
                nbins=30,
                barmode="overlay",
                marginal="box",
                color_discrete_map={0: "green", 1: "red"},
                title=f"{feature} Distribution by Heart Disease"
            )

            fig_2 = px.violin(self.df, y=feature, x="HeartDisease", color="HeartDisease", box=True,
                title=f"{feature} Distribution by Heart Disease")
            

            fig_1.update_layout(xaxis_title=feature, yaxis_title="Count")
            fig_2.update_layout(xaxis_title="Heart Disease", yaxis_title=feature)
            fig_1.update_layout(legend_title_text="Heart Disease")
            fig_2.update_layout(legend_title_text="Heart Disease")

            st.plotly_chart(fig_1, use_container_width=True)
            st.plotly_chart(fig_2, use_container_width=True)


    def model_insights(self):
    

        st.markdown("## 🔢 Feature Importance & Confusion Matrices")

        # Load models
        rf_model = joblib.load("model/random_forest.pkl")
        xgb_model = joblib.load("model/xgboost.pkl")
        lr_model = joblib.load("model/logistic_regression.pkl")

        # Load test data
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv").values.ravel()

        expected_cols = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
            'ST_Slope_Flat', 'ST_Slope_Up'
        ]

        if list(X_test.columns) != expected_cols:
            X_test = X_test[expected_cols]

        models = {
            "Logistic Regression": lr_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
        }

        for name, model in models.items():
            st.markdown(f"### 📌 {name}")

            # -- Confusion Matrix
            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{name} - Confusion Matrix")
            st.pyplot(fig_cm)

            # -- Feature Importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                imp_df = pd.DataFrame({
                    "Feature": expected_cols,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)

                fig_fi = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    title=f"{name} - Feature Importance"
                )
                st.plotly_chart(fig_fi)

            elif name == "Logistic Regression":
                coefs = model.coef_[0]
                coef_df = pd.DataFrame({
                    "Feature": expected_cols,
                    "Importance": np.abs(coefs),
                    "Coefficient": coefs
                }).sort_values("Importance", ascending=False)

                fig_coef = px.bar(
                    coef_df, x="Coefficient", y="Feature", orientation="h",
                    title="Logistic Regression - Coefficients (Signed)"
                )
                st.plotly_chart(fig_coef)

            # -- SHAP Summary Plot
            shap_path = f"plots/shap_{name.lower().replace(' ', '_')}.png"
            try:
                st.image(shap_path, caption=f"{name} - SHAP Summary", use_container_width=True)
            except:
                st.warning(f"SHAP plot not found for {name}. Save it at `{shap_path}`.")

            st.markdown("---")

            


    






    def render(self):
        st.title("📊 Patient Risk Insights Dashboard")
        self.heart_disease_distribution()
        self.age_distribution()
        self.categorical_vs_target()
        self.numerical_vs_target()
        self.model_insights()

