# Customer Churn Prediction Using LSTM and Attention Networks

## Overview

Customer churn prediction is an important problem in the telecommunications industry. Companies lose revenue when customers discontinue their services. This project uses **Deep Learning with LSTM and an Attention Mechanism** to predict whether a customer is likely to churn.

The system also integrates **Explainable AI using SHAP**, allowing users to understand which features influenced the model’s prediction.

A **Streamlit-based web application** is built to allow users to input customer details and receive real-time predictions along with feature explanations.

---

## Features

* Deep learning model using **LSTM (Long Short-Term Memory)**
* Custom **Attention Mechanism** to focus on important features
* **Explainable AI using SHAP** to interpret predictions
* **Interactive Streamlit Web App**
* Customer risk classification (Low / Medium / High)
* Feature importance visualization using SHAP plots

---

## Project Architecture

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
LSTM + Attention Model Training
   │
   ▼
Model Saved (.pkl)
   │
   ▼
Streamlit Web Application
   │
   ▼
User Input → Prediction → SHAP Explanation
```

---

## Project Structure

```
Customer-Churn-Prediction-Using-LSTM-and-Attention-Networks
│
├── data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models
│   └── hix_lstm_attnxai.pkl
│
├── src
│   ├── attention_layer.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── shap_explainer.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **Telco Customer Churn Dataset**, which contains customer information such as:

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Internet Service
* Online Security
* Tech Support
* Payment Method

Target variable:

```
Churn
Yes → Customer leaves
No → Customer stays
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* SHAP (Explainable AI)
* Streamlit (Web UI)
* Matplotlib

---

## Model Architecture

Input Layer
↓
LSTM Layer (64 Units)
↓
Attention Layer
↓
Dense Layer (32 Units)
↓
Sigmoid Output Layer

The model outputs the **probability of customer churn**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction-Using-LSTM-and-Attention-Networks.git
cd Customer-Churn-Prediction-Using-LSTM-and-Attention-Networks
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script:

```bash
python src/train_model.py
```

This will:

* Train the LSTM-Attention model
* Save the trained model to:

```
models/hix_lstm_attnxai.pkl
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## How the System Works

1. User enters customer details in the Streamlit interface.
2. Input features are scaled using the trained scaler.
3. Data is reshaped for LSTM input.
4. The trained model predicts churn probability.
5. Risk level is categorized:

   * Low Risk
   * Medium Risk
   * High Risk
6. SHAP generates feature explanations.
7. Results and visualizations are displayed.

---

## Explainable AI (SHAP)

SHAP (SHapley Additive Explanations) is used to explain model predictions by showing how each feature contributes to the final prediction.

Example explanation output:

| Feature        | Impact |
| -------------- | ------ |
| MonthlyCharges | +0.31  |
| Tenure         | -0.22  |
| Contract       | -0.10  |

Positive values increase churn probability, while negative values decrease it.

---

## Example Output

```
Churn Probability: 0.72

Risk Level: HIGH

Top Factors Influencing Prediction:
- Monthly Charges
- Contract Type
- Tenure
```

---

## Future Improvements

* Add model evaluation metrics (Accuracy, Precision, Recall, F1-score)
* Deploy the application using Docker
* Improve feature engineering
* Add support for multiple datasets
* Implement attention heatmap visualization

---

## Author

Lavudya Sai Nikhil

B.Tech Computer Science and Engineering
Specialization: Artificial Intelligence and Machine Learning

---

## License

This project is open-source and available under the MIT License.
