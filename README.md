# Stock Price Prediction with LSTM (Python + TensorFlow/Keras)

This project is a Python-based implementation of a **stock price prediction model** using **LSTM (Long Short-Term Memory) neural networks**. The model predicts the **next closing price** of a stock (e.g., Apple, AAPL) based on historical OHLCV (Open, High, Low, Close, Volume) data.

---

## Requirements

You can install required packages via:

```bash
pip install -r requirements.txt
```

`requirements.txt` example:

```
pandas
numpy
scikit-learn
yfinance
tensorflow
```

---


## Usage

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/tf-stock-predictor.git
cd tf-stock-predictor
```

2. **Run the main script**

```bash
python main.py
```

- The script will **download historical data** (if CSV does not exist), preprocess it, create sequences, and either load an existing model or train a new one.  
- Model training uses **early stopping** and **model checkpointing** to save the best model automatically.  

3. **Output**

- The script prints the **predicted next closing price** (scaled 0–1).  
- The trained model is saved as `lstm_aapl_model.keras`.  
