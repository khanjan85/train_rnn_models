import os
import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Paths
DB_PATH = "nifty50_data_v1.db"
PREDICTION_DB_PATH = "prediction.db"
MODELS_DIR = "models"

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def get_tables(database_path):
    """
    Retrieve table names from the database excluding sqlite_sequence.

    Args:
        database_path (str): Path to the SQLite database.
    Returns:
        list: List of table names.
    """
    with sqlite3.connect(database_path) as conn:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence';"
        tables = pd.read_sql(query, conn)['name'].tolist()
    return tables

def preprocess_data(data, feature_columns, target_columns, look_back=12):
    """
    Prepare the data for time series prediction.

    Args:
        data (pd.DataFrame): Input data with Datetime and required columns.
        feature_columns (list): List of feature column names.
        target_columns (list): List of target column names.
        look_back (int): Number of past steps to consider for prediction.
    Returns:
        tuple: Scaled features and targets as numpy arrays, along with scalers.
    """
    # Drop duplicate rows by Datetime
    data = data[~data.index.duplicated(keep='first')]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_columns + target_columns])

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back, :len(feature_columns)])
        y.append(scaled_data[i + look_back, len(feature_columns):])
    return np.array(X), np.array(y), scaler

def train_rnn_model(X_train, y_train, input_shape):
    """
    Train an enhanced RNN model on the given data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        input_shape (tuple): Shape of the input data.
    Returns:
        keras.Model: Trained RNN model.
    """
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(y_train.shape[1])  # Output layer matches target columns
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
    return model

def main():
    """
    Main function to train RNN models and store predictions.
    """
    tables = get_tables(DB_PATH)
    feature_columns = ["Open", "High", "Low", "Close", "Volume"]
    target_columns = ["Open", "High", "Low", "Close", "Volume"]
    look_back = 12  # 12 steps back (5 mins each = 60 mins)

    with sqlite3.connect(DB_PATH) as conn, sqlite3.connect(PREDICTION_DB_PATH) as pred_conn:
        for table in tables:
            print(f"Processing table: {table}")
            data = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Datetime"])
            data.set_index("Datetime", inplace=True)

            # Prepare train-test split
            train_size = int(len(data) * 0.8)
            train_data, test_data = data[:train_size], data[train_size:]

            # Preprocess data
            X_train, y_train, scaler = preprocess_data(train_data, feature_columns, target_columns, look_back)
            X_test, y_test, _ = preprocess_data(test_data, feature_columns, target_columns, look_back)

            # Train model
            model = train_rnn_model(X_train, y_train, X_train.shape[1:])

            # Save model and scaler
            model.save(os.path.join(MODELS_DIR, f"{table}_model.h5"))
            dump(scaler, os.path.join(MODELS_DIR, f"{table}_scaler.joblib"))

            # Make predictions
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(np.hstack((np.zeros_like(predictions), predictions)))[:, len(feature_columns):]
            true_values = scaler.inverse_transform(np.hstack((np.zeros_like(y_test), y_test)))[:, len(feature_columns):]

            # Align predictions with Datetime index
            datetime_index = test_data.index[look_back:].to_list()
            result_df = pd.DataFrame({
                "Datetime": datetime_index,
                **{f"Pred_{col}": predictions[:, idx] for idx, col in enumerate(target_columns)},
                **{f"True_{col}": true_values[:, idx] for idx, col in enumerate(target_columns)}
            })

            # Save predictions to database
            result_df.to_sql(table, pred_conn, if_exists="replace", index=False)
            print(f"Finished processing table: {table}")

if __name__ == "__main__":
    main()
