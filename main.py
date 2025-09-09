import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import tensorflow as tf

from config import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_TICKER,
    FEATURES,
    MODEL_FILENAME,
    SEQUENCE_LENGTH,
)
from data.data_loader import load_data, resample_to_10m
from data.preprocess import create_sequences, scale_ohlcv, train_val_test_split
from data.validators import validate_data, validate_scaled_data
from model.lstm import build_lstm_model

# 1. Load data
raw_data = load_data(
    DEFAULT_TICKER, DEFAULT_INTERVAL, DEFAULT_PERIOD, force_download=True
)
data_10m = resample_to_10m(raw_data)
validate_data(data_10m)

# 2. Preprocess
scaled_data, scaler_close = scale_ohlcv(data_10m)
validate_scaled_data(scaled_data)

train_data, val_data, test_data = train_val_test_split(scaled_data)
X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
X_val, y_val = create_sequences(val_data, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)

# 3. Train or load model
if os.path.exists(MODEL_FILENAME):
    model = tf.keras.models.load_model(MODEL_FILENAME)
else:
    model = build_lstm_model(SEQUENCE_LENGTH, len(FEATURES))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_FILENAME, monitor="val_loss", save_best_only=True
            ),
        ],
    )

# 4. Evaluate
print("Test loss:", model.evaluate(X_test, y_test))

# 5. Predict
last_seq = scaled_data[-SEQUENCE_LENGTH:].values.reshape(
    1, SEQUENCE_LENGTH, len(FEATURES)
)
pred = model.predict(last_seq)[0][0]
real_pred = scaler_close.inverse_transform([[pred]])[0][0]
print(
    f"Prediction for {data_10m.index[-1] + pd.Timedelta(minutes=15)}: {real_pred:.2f} USD"
)
