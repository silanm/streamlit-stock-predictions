import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# Streamlit App Title
st.title("Financial Forecasting App")

# Sidebar Parameters
st.sidebar.header("Adjustable Parameters")

# Select stock ticker
stock_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")

# Select historical data range
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("2023-01-01"))

# API key for RapidAPI
rapidapi_key = st.sidebar.text_input("Enter X-RapidAPI-Key:", type="password")

# Model parameters
epochs = st.sidebar.slider("Training Epochs:", 1, 50, 10)
lstm_layers = st.sidebar.slider("LSTM Layers:", 1, 3, 1)
batch_size = st.sidebar.slider("Batch Size:", 8, 128, 32)

# Cache historical data retrieval to improve performance
@st.cache_data
def fetch_data_from_rapidapi(ticker, start, end, api_key):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }
    params = {"symbol": ticker}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        prices = data.get("prices", [])
        historical_data = [
            {
                "Date": pd.to_datetime(item["date"], unit="s"),
                "Close": item.get("close", None)
            }
            for item in prices if "close" in item
        ]
        df = pd.DataFrame(historical_data)
        df.set_index("Date", inplace=True)
        df = df.loc[start:end]
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load and process historical data
st.subheader("Historical Stock Data")
if rapidapi_key:
    data = fetch_data_from_rapidapi(stock_ticker, start_date, end_date, rapidapi_key)
    if data is not None and not data.empty:
        st.write(data.tail())
        # Plot historical data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"{stock_ticker} Stock Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    else:
        st.error("No data found for the selected ticker and date range.")
else:
    st.warning("Please enter your X-RapidAPI-Key.")

# Prepare data for LSTM
if rapidapi_key and data is not None and not data.empty:
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create training dataset
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=(lstm_layers > 1), input_shape=(X_train.shape[1], 1)))
    for _ in range(lstm_layers - 1):
        model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    if st.sidebar.button("Train Model"):
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        st.success("Model training complete!")

    # Predict future prices
    st.subheader("Predicted Stock Prices")
    test_data = scaled_data[train_size - time_step:]
    X_test, y_test = create_dataset(test_data, time_step)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=actual_prices.flatten(), mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted Prices'))
    fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Download predicted data
    result_df = pd.DataFrame({
        "Date": data.index[-len(predictions):],
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predictions.flatten()
    })
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name=f"{stock_ticker}_predictions.csv", mime="text/csv")

# How it works section
with st.expander("How it Works"):
    st.write("This app fetches historical stock data from Yahoo Finance via RapidAPI and trains a Long Short-Term Memory (LSTM) model to predict stock prices for the upcoming week. You can adjust parameters such as epochs, LSTM layers, and batch size to customize the prediction.")
    st.write("The predicted prices and historical prices are visualized interactively using Plotly.")