import os
import sqlite3
import pandas as pd
import tensorflow as tf

DATABASE_PATH = "nifty50_data_v1.db"
PREDICTIONS_DB = "predictions/predictions.db"  # Ensure a directory path is included
MODELS_FOLDER = "models"

# Ensure the folder for PREDICTIONS_DB exists
os.makedirs(os.path.dirname(PREDICTIONS_DB), exist_ok=True)

# Connect to the database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Load trained models and make predictions for each table
for table_name in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    table_name = table_name[0]

    if table_name == "sqlite_sequence":
        continue

    # Load data
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn)

    # Preprocess data
    features = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]
    data = data[features]

    # Load model
    model_path = os.path.join(MODELS_FOLDER, f"{table_name}_rnn_model.h5")
    if not os.path.exists(model_path):
        print(f"Model for table {table_name} not found. Skipping...")
        continue

    model = tf.keras.models.load_model(model_path)

    # Make predictions
    X_test = data[-12:].values.reshape(1, 12, len(features))  # Last 12 steps for prediction
    predictions = model.predict(X_test)

    # Save predictions to a new database
    predictions_df = pd.DataFrame(predictions, columns=features)
    predictions_df["Datetime"] = data["Datetime"].iloc[-12:].values
    predictions_df["Actual"] = data.iloc[-12:][features].values.tolist()

    predictions_conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(table_name, predictions_conn, if_exists="replace", index=False)
    predictions_conn.close()

conn.close()
