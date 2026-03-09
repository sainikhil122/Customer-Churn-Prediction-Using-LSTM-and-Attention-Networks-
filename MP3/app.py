

import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTANT: import custom layer BEFORE loading model
from src.attention_layer import HiXAttention

st.set_page_config(page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction")

# Load model
with open("models/hix_lstm_attnxai.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["features"]

st.subheader("Customer Details")

# tenure = st.number_input("Tenure (months)", 0, 100)

# monthly = st.number_input("Monthly Charges")

# total = st.number_input("Total Charges")
tenure = st.text_input("Tenure (months)", placeholder="Enter months")

monthly = st.text_input("Monthly Charges", placeholder="Enter monthly charges")

total = st.text_input("Total Charges", placeholder="Enter total charges")

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet = st.selectbox("Internet Service", ["No", "Yes"])
internet = 0 if internet == "No" else 1

security = st.selectbox("Online Security", ["No", "Yes"])
security = 0 if security == "No" else 1

tech = st.selectbox("Tech Support", ["No", "Yes"])
tech = 0 if tech == "No" else 1

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

if st.button("Predict Churn"):

    try:
        tenure = float(tenure)
        monthly = float(monthly)
        total = float(total)
    except:
        st.error("Please enter valid numeric values")
        st.stop()

    user = [
        tenure,
        monthly,
        total,
        contract_map[contract],
        internet,
        security,
        tech,
        payment_map[payment]
    ]

    user = np.array(user).reshape(1, -1)

    user_scaled = scaler.transform(user)

    user_seq = user_scaled.reshape(1, user.shape[1], 1)

    prob = float(model.predict(user_seq)[0][0])
    
    st.write("Model Probability:", prob)

    st.subheader("Prediction")

    st.write(f"Churn Probability: {round(prob,3)}")

    # Risk level logic
    if prob < 0.30:
        risk = "LOW RISK"

        st.markdown(
            f"""
            <div style="
                background-color:#e6f4ea;
                padding:15px;
                border-radius:10px;
                border-left:8px solid #2e7d32;
                font-size:18px;
                font-weight:600;
                color:#1b5e20;">
                ✅ Customer Not Likely to Churn<br>
                Risk Level: {risk}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif prob < 0.60:
        risk = "MEDIUM RISK"

        st.markdown(
            f"""
            <div style="
                background-color:#fff8e1;
                padding:15px;
                border-radius:10px;
                border-left:8px solid #f9a825;
                font-size:18px;
                font-weight:600;
                color:#e65100;">
                ⚠️ Customer May Churn<br>
                Risk Level: {risk}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        risk = "HIGH RISK"

        st.markdown(
            f"""
            <div style="
                background-color:#fdecea;
                padding:15px;
                border-radius:10px;
                border-left:8px solid #c62828;
                font-size:18px;
                font-weight:600;
                color:#b71c1c;">
                🚨 Customer Likely to Churn<br>
                Risk Level: {risk}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------

    st.subheader("SHAP Explanation")

    def shap_predict(X):

        X_scaled = scaler.transform(X)

        X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        return model.predict(X_seq)

    background = np.random.normal(size=(50, len(feature_names)))

    explainer = shap.KernelExplainer(shap_predict, background)

    shap_values = explainer.shap_values(user)

    shap_vals = np.array(shap_values).flatten()

    features = user.flatten()

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        data=features,
        feature_names=feature_names
    )

    fig, ax = plt.subplots()

    shap.plots.waterfall(explanation, show=False)

    st.pyplot(fig)

    # -----------------------------
    # SHAP INTERPRETATION
    # -----------------------------

    # st.subheader("SHAP Graph Interpretation")

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": features,
        "SHAP Value": shap_vals
    })

    increase = shap_df[shap_df["SHAP Value"] > 0]
    decrease = shap_df[shap_df["SHAP Value"] < 0]

    increase = increase.sort_values("SHAP Value", ascending=False)
    decrease = decrease.sort_values("SHAP Value")

    st.markdown("### Top Factors Increasing Churn")

    for _, row in increase.head(3).iterrows():
        st.write(f"• {row['Feature']}")

    st.markdown("### Top Factors Reducing Churn")

    for _, row in decrease.head(3).iterrows():
        st.write(f"• {row['Feature']}")