import numpy as np
import pandas as pd
import shap


def predict_customer(model, scaler, explainer, feature_names, user):

    user = np.array(user).reshape(1,-1)

    user_scaled = scaler.transform(user)

    user_seq = user_scaled.reshape(1,user.shape[1],1)

    prob = float(model.predict(user_seq)[0][0])

    if prob < 0.30:
        risk = "LOW RISK"
    elif prob < 0.60:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    shap_values = explainer.shap_values(user)

    shap_vals = shap_values[0].flatten()

    shap_df = pd.DataFrame({
        "Feature":feature_names,
        "SHAP_Value":shap_vals
    })

    shap_df["Impact"] = shap_df["SHAP_Value"].abs()

    shap_df = shap_df.sort_values("Impact",ascending=False)

    return prob, risk, shap_values, shap_df