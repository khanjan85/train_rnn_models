import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Constants
DB_URL = "https://raw.githubusercontent.com/chiragpalan/stocks_data_management/main/nifty50_data_v1.db"
LOCAL_DB = "nifty50_data_v1.db"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
PREDICTIONS_DB = os.path.join(PREDICTIONS_DIR, "predictions_v1.db")

# Step 1: Download database
def download_database():
    import requests
    response = requests.get(DB_URL)
    with open(LOCAL_DB, "wb") as file:
        file.write(response.content)

# Step 2: Prepare data
def prepare_data(df):
    # Remove duplicates based on Datetime column
    df = df.drop_duplicates(subset="Datetime")
    # Sort by Datetime for consistent order
    df = df.sort_values(by="Datetime")
    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
    return scaled_data, scaler

# Step 3: Create sliding windows for RNN
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i : i + input_steps])
        y.append(data[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)

# Step 4: Build and train RNN model
def build_rnn_model(input_shape, output_size):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(output_size)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Step 5: Train models and save outputs
def train_and_save_models():
    # Create directories for outputs
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(LOCAL_DB)
    predictions_conn = sqlite3.connect(PREDICTIONS_DB)

    # Loop through each table in the database
    for table in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"]:
        print(f"Processing table: {table}")
        
        # Load and prepare data
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        data, scaler = prepare_data(df)
        
        # Create input and output sequences
        input_steps, output_steps = 24, 12
        X, y = create_sequences(data, input_steps, output_steps)
        y = y.reshape(y.shape[0], -1)  # Flatten output
        
        # Train the RNN model
        model = build_rnn_model(X.shape[1:], y.shape[1])
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        
        # Save the model and scaler
        model_path = os.path.join(MODELS_DIR, f"{table}_rnn_model.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{table}_scaler.pkl")
        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save predictions to the database
        predictions = model.predict(X)
        predictions_df = pd.DataFrame(predictions.reshape(-1, 12, 5).mean(axis=1), columns=["Open", "High", "Low", "Close", "Volume"])
        predictions_df["Datetime"] = df["Datetime"].iloc[input_steps : input_steps + len(predictions)].values
        predictions_df.to_sql(f"{table}_predictions", predictions_conn, if_exists="replace", index=False)

    conn.close()
    predictions_conn.close()

# Main execution
if __name__ == "__main__":
    download_database()
    train_and_save_models()
