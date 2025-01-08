import os
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

DATABASE_PATH = "nifty50_data_v1.db"
PREDICTIONS_DB = "predictions.db"
MODELS_FOLDER = "models"
os.makedirs(os.path.dirname(PREDICTIONS_DB), exist_ok=True)

def predict_table(table_name):
    # Load model and scaler
    model = load_model(f"{MODELS_FOLDER}/{table_name}_model.h5")
    with open(f"{MODELS_FOLDER}/{table_name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    conn = sqlite3.connect(DATABASE_PATH)
    data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Prepare input
    test_data = scaler.transform(data)
    X_test = []
    for i in range(12, len(test_data)):
        X_test.append(test_data[i-12:i])
    X_test = np.array(X_test)

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Save predictions
    prediction_df = pd.DataFrame(predictions, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    prediction_df['Datetime'] = data['Datetime'].iloc[12:].values
    prediction_df['Actual_Open'] = data['Open'].iloc[12:].values
    prediction_df['Actual_High'] = data['High'].iloc[12:].values
    prediction_df['Actual_Low'] = data['Low'].iloc[12:].values
    prediction_df['Actual_Close'] = data['Close'].iloc[12:].values
    prediction_df['Actual_Volume'] = data['Volume'].iloc[12:].values

    conn = sqlite3.connect(PREDICTIONS_DB)
    prediction_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Predictions saved for table: {table_name}")

# Predict for all tables
conn = sqlite3.connect(DATABASE_PATH)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
conn.close()

for table in tables['name']:
    if table != 'sqlite_sequence':
        predict_table(table)
