import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Connect to the database
DATABASE_PATH = "nifty50_data_v1.db"
MODELS_FOLDER = "models"
os.makedirs(MODELS_FOLDER, exist_ok=True)

def load_data(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_rnn_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(5)  # Predicting 5 features: Open, High, Low, Close, Volume
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(table_name):
    print(f"Processing table: {table_name}")
    data = load_data(table_name)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Split into train, validation, and test
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    train, val, test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]

    # Preprocess
    train_scaled, scaler = preprocess_data(train)
    X_train, y_train = [], []
    for i in range(12, len(train_scaled)):
        X_train.append(train_scaled[i-12:i])
        y_train.append(train_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Train model
    model = create_rnn_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save model and scaler
    model_path = f"{MODELS_FOLDER}/{table_name}_model.h5"
    file_path = f"{MODELS_FOLDER}/{table_name}_scaler.pkl"
    print("model path", model_path)
    if os.path.exists(model_path):
        print("removing models")
        os.remove(model_path) 
    print(f"Saving scaler to: {file_path}")
    if os.path.exists(file_path):
        print("removing files")
        os.remove(file_path)
        
    print(f"Existing file {file_path} deleted.")

    model.save(model_path)
    print(f"Model for {table_name} saved.")
    with open(file_path, "wb") as f:
        pickle.dump(scaler, f)
    

# List all tables
conn = sqlite3.connect(DATABASE_PATH)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
conn.close()

# Train models for each table
for table in tables['name']:
    if table != 'sqlite_sequence':
        train_model(table)
