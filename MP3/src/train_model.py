import numpy as np
import pickle
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from preprocessing import load_data
from attention_layer import HiXAttention


X_scaled, y, scaler, feature_names = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1],1)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, stratify=y, random_state=42
)

inputs = Input(shape=(X_train.shape[1],1))

x = LSTM(64, return_sequences=True)(inputs)

context, attention_weights = HiXAttention()(x)

x = Dense(32,activation='relu')(context)

output = Dense(1,activation='sigmoid')(x)

model = Model(inputs,output)

model.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test,y_test)
)

with open("models/hix_lstm_attnxai.pkl","wb") as f:

    pickle.dump({
        "model":model,
        "scaler":scaler,
        "features":feature_names
    },f)