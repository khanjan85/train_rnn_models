# train_rnn_models
This repo trains RNN models on stock data

# Train RNN Models

This repository contains a pipeline to train RNN models for time-series forecasting on stock data.

## Setup Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/chiragpalan/train_rnn_models.git
    cd train_rnn_models
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Fetch the database:
    ```bash
    python fetch_db.py
    ```

4. Train the models:
    ```bash
    python train_rnn.py
    ```

## GitHub Actions Workflow

Every push to the `main` branch triggers a workflow to:
1. Fetch the database.
2. Train the RNN models.
3. Store models and predictions as artifacts.

## Outputs
- **Trained Models**: Stored in the `models/` directory.
- **Predictions**: Stored in `prediction.db`.
