# 🫀 Heart Failure Risk Predictor & Insights App

A Streamlit-powered interactive application that predicts the risk of heart failure using patient health metrics. It incorporates machine learning models (Logistic Regression, Random Forest, and XGBoost) and presents probability-based predictions, visual explanations, and feature insights.

---

## 📌 Project Highlights

- 🔢 **User Form** for entering patient health data
- 🧠 **Predictions** from 3 trained ML models
- 📈 **Model Evaluation Metrics**: Accuracy, Precision, Recall, ROC-AUC
- 🟨 **Confusion Matrices** for interpretability
- 🧩 **Feature Importance & SHAP Visuals** (for Random Forest and XGBoost)
- 📊 **EDA Dashboard** with interactive insights and visualizations

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **ML Models**: scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn, SHAP
- **Data**: UCI Heart Disease Dataset

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yAkpere38/heart-failure-predictor.git
cd heart-failure-predictor

pip install -r requirements.txt
streamlit run app.py
