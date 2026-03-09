import shap


def build_explainer(model, scaler, X):

    def shap_predict(X_flat):

        X_scaled = scaler.transform(X_flat)

        X_seq = X_scaled.reshape(X_scaled.shape[0],X_scaled.shape[1],1)

        return model.predict(X_seq)

    explainer = shap.KernelExplainer(shap_predict, X[:100])

    return explainer