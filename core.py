import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

MODEL_FILENAME = "lstm_aapl_model.keras"
SEQUENCE_LENGTH = 20  # number of previous candles to use for prediction
FEATURES = ["Open", "High", "Low", "Close", "Volume"]  # input features


print("TensorFlow version:", tf.__version__)


def load_data(ticker="AAPL", interval="10m", period="3mo", filename=None):
    if filename is None:
        filename = f"{ticker}_{interval}_{period}.csv"

    if os.path.exists(filename):
        print(f"Loading data from a file: {filename}")
        data = pd.read_csv(
            filename,
            skiprows=3,
            names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
            index_col=0,
            parse_dates=True,
        )
    else:
        print(f"Downloading data {ticker} ({interval}, {period})...")
        data = yf.download(
            ticker,
            interval=interval,
            period=period,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]

        data.to_csv(filename)
        print(f"Data saved to file: {filename}")

    return data


def validate_data(data):
    valid = True  # flag to track if all checks pass

    # Check for missing values
    if data.isnull().values.any():
        print("Missing values detected!")
        print(data.isnull().sum())
        valid = False
    else:
        print("No missing values found")

    # Check for negative or zero prices (Open, High, Low, Close)
    price_cols = ["Open", "High", "Low", "Close"]
    if (data[price_cols] <= 0).any().any():
        print("Negative or zero prices detected!")
        print(data[data[price_cols] <= 0])
        valid = False
    else:
        print("All prices are positive")

    # Check for negative volume
    if (data["Volume"] < 0).any():
        print("Negative volume detected!")
        print((data["Volume"] < 0))
        valid = False
    else:
        print("Volume values are valid")

    return valid


ticker = "AAPL"

data_5m = load_data("AAPL", interval="5m", period="60d")

data_10m = (
    data_5m.resample("10min")
    .agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    .dropna()
)

print(f"Validate csv data:{'OK' if validate_data(data_10m) else 'BAD'}")


def validate_scaled_data(scaled_data):
    valid = True

    # Check for NaN values
    if scaled_data.isnull().values.any():
        print("NaN values detected in scaled data!")
        valid = False

    # Check for infinite values
    if np.isinf(scaled_data.values).any():
        print("Infinite values detected in scaled data!")
        valid = False

    # Check if any values are outside the expected range [0,1]
    if (scaled_data < 0).any().any() or (scaled_data > 1).any().any():
        print("Some values are out of the expected range [0,1]")
        valid = False
    else:
        print("All values are within the expected range [0,1]")

    return valid


def scale_ohlcv(data):
    price_cols = ["Open", "High", "Low", "Close"]

    prices = data[price_cols].copy()
    volume = data[["Volume"]].copy()

    scaler_prices = MinMaxScaler(feature_range=(0, 1))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))

    prices_scaled = scaler_prices.fit_transform(prices)
    volume_scaled = scaler_volume.fit_transform(np.log1p(volume))
    scaler_close.fit_transform(prices[["Close"]])

    scaled_df = pd.DataFrame(
        np.hstack([prices_scaled, volume_scaled]),
        columns=price_cols + ["Volume"],
        index=data.index,
    )

    return scaled_df, scaler_close


scaled_data, scaler_close = scale_ohlcv(data_10m)
print(
    f"Validate scaled data result: {'OK' if validate_scaled_data(scaled_data) else 'BAD'}"
)


def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i : i + seq_length].values)
        y.append(data.iloc[i + seq_length]["Close"])
    return np.array(X), np.array(y)


def train_val_test_split(data, train_ratio=0.7, val_ratio=0.1):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    return train_data, val_data, test_data


def build_lstm_model(seq_length, num_features):
    inputs = tf.keras.layers.Input(
        shape=(seq_length, num_features), name="OHLCV_input"
    )

    # First LSTM layer with 64 units, returns full sequence for next LSTM
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(
        x
    )  # regularization to prevent overfitting

    # Second LSTM layer with 32 units, returns only last output
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output layer: predict the next closing price
    outputs = tf.keras.layers.Dense(1, name="next_close")(x)

    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model for regression
    model.compile(optimizer="adam", loss="mse")

    return model


train_data, val_data, test_data = train_val_test_split(scaled_data)

X_train, y_train = create_sequences(train_data)
X_val, y_val = create_sequences(val_data)
X_test, y_test = create_sequences(test_data)

# 3. Load or train model
if os.path.exists(MODEL_FILENAME):
    print(f"Loading existing model: {MODEL_FILENAME}")
    model = tf.keras.models.load_model(MODEL_FILENAME)
else:
    print("Training new LSTM model...")
    model = build_lstm_model(
        seq_length=SEQUENCE_LENGTH, num_features=len(FEATURES)
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILENAME, monitor="val_loss", save_best_only=True
    )

    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

# 4. Evaluate on test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE loss: {test_loss:.6f}")

# 5. Predict next closing price
last_sequence = scaled_data[-SEQUENCE_LENGTH:].values.reshape(
    1, SEQUENCE_LENGTH, len(FEATURES)
)
predicted_next_close = model.predict(last_sequence)
print(
    f"Predicted next closing price (scaled 0-1): {predicted_next_close[0][0]}"
)


predicted_next_close = model.predict(last_sequence)
scaled_pred = predicted_next_close[0][0]

real_pred = scaler_close.inverse_transform([[scaled_pred]])[0][0]
predicted_time = data_10m.index[-1] + pd.Timedelta(minutes=15)

print(f"Prediction for {predicted_time} in USD: {real_pred:.2f}")
