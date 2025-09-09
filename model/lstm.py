import tensorflow as tf


def build_lstm_model(
    seq_length: int,
    num_features: int,
    lstm_units: tuple = (64, 32),
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile an LSTM model for time series forecasting.

    Architecture:
        - Input: sequence of OHLCV features
        - LSTM layers with dropout
        - Dense output layer (regression)

    Args:
        seq_length (int): Number of timesteps in each input sequence.
        num_features (int): Number of features per timestep.
        lstm_units (tuple): Number of units in each LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """

    # Input layer
    inputs = tf.keras.layers.Input(
        shape=(seq_length, num_features), name="OHLCV_input"
    )

    # First LSTM layer (returns sequences for stacking)
    x = tf.keras.layers.LSTM(lstm_units[0], return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Second LSTM layer (returns final hidden state)
    x = tf.keras.layers.LSTM(lstm_units[1])(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer: regression to predict next Close price
    outputs = tf.keras.layers.Dense(1, name="next_close")(x)

    # Build model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model
