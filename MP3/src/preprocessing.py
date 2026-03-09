import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path):

    df = pd.read_csv(path)

    df = df[['tenure','MonthlyCharges','TotalCharges',
             'Contract','InternetService','OnlineSecurity',
             'TechSupport','PaymentMethod','Churn']]

    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    df.dropna(inplace=True)

    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

    df['Contract'] = df['Contract'].map({
        'Month-to-month':0,
        'One year':1,
        'Two year':2
    })

    df['InternetService'] = df['InternetService'].map({
        'No':0,'Yes':1,'Fiber optic':1
    })

    df['OnlineSecurity'] = df['OnlineSecurity'].map({'No':0,'Yes':1})
    df['TechSupport'] = df['TechSupport'].map({'No':0,'Yes':1})

    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check':0,
        'Mailed check':1,
        'Bank transfer (automatic)':2,
        'Credit card (automatic)':3
    })

    feature_names = df.drop('Churn',axis=1).columns.tolist()

    X = df.drop('Churn',axis=1).values
    y = df['Churn'].values.astype(float)

    X = np.nan_to_num(X)

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names