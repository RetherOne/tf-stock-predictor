import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model


def load_data(ticker="AAPL", interval="10m", period="3mo", filename=None):
    if filename is None:
        filename = f"{ticker}_{interval}_{period}.csv"

    if os.path.exists(filename):
        print(f"✅ Загружаем данные из файла: {filename}")
        data = pd.read_csv(
            filename,
            skiprows=3,
            names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
            index_col=0,
            parse_dates=True,
        )
    else:
        print(f"📥 Скачиваем данные {ticker} ({interval}, {period})...")
        data = yf.download(
            ticker,
            interval=interval,
            period=period,
        )
        data.to_csv(filename)
        print(f"💾 Данные сохранены в файл: {filename}")

    return data


ticker = "AAPL"
# Скачиваем 5-минутки
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

print(data_10m.head())

features = ["Open", "High", "Low", "Close", "Volume"]
data_features = data_10m[features].copy()


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_features)


L = 60
H = 1

X = []
y = []

for i in range(L, len(data_scaled) - H + 1):
    X.append(data_scaled[i - L : i])
    y.append(data_scaled[i + H - 1][3])  # Close

X = np.array(X)
y = np.array(y)

print("X.shape:", X.shape)
print("y.shape:", y.shape)


train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size : train_size + val_size]
y_val = y[train_size : train_size + val_size]

X_test = X[train_size + val_size :]
y_test = y[train_size + val_size :]

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


model_filename = "lstm_model_10m.h5"

if os.path.exists(model_filename):
    print("✅ Модель уже обучена. Загружаем с диска...")
    model = load_model(model_filename)
else:
    print("📦 Обучаем новую модель...")

    model = Sequential(
        [
            LSTM(
                50,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            ),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(model_filename)
    print(f"💾 Модель сохранена в файл: {model_filename}")

y_pred = model.predict(X_test)
print("Test predictions shape:", y_pred.shape)
