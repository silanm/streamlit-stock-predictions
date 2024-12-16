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
st.title("Stock Prices Forecasting App")

# Sidebar Parameters
st.sidebar.header("Adjustable Parameters")

# API key for Financial Modeling Prep
api_key = st.sidebar.text_input("Enter Financial Modeling Prep API Key:", type="password")
# 5X4d3etoabPqTH9MSMwtaKN0fff3eTmE
# https://site.financialmodelingprep.com/developer/docs/dashboard

# Select stock ticker
stock_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")

# Select historical data range
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("2024-12-31"))

# Model parameters
epochs = st.sidebar.slider("Training Epochs:", 1, 50, 10) # Number of times the model will be trained on the dataset
lstm_layers = st.sidebar.slider("LSTM Layers:", 1, 3, 1)  # Number of LSTM layers in the model, more layers can capture more complex patterns
batch_size = st.sidebar.slider("Batch Size:", 8, 128, 32) # Number of samples processed before the model is updated 

# Cache historical data retrieval to improve performance
@st.cache_data
def fetch_data_from_fmp(ticker, start, end, api_key):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "apikey": api_key
    } # API parameters

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for 4xx/5xx status codes
        data = response.json() # Convert response to JSON
        if "historical" in data:
            historical_data = [
                {
                    "Date": pd.to_datetime(item["date"]),
                    "Close": item.get("close", None)
                }
                for item in data["historical"]
            ]
            df = pd.DataFrame(historical_data)
            df.set_index("Date", inplace=True)
            return df
        else:
            st.error("No historical data found.")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load and process historical data
st.subheader("Historical Stock Data")
if api_key:
    data = fetch_data_from_fmp(stock_ticker, start_date, end_date, api_key)
    if data is not None and not data.empty:
        st.write(data.head(20)) # Display first 5 rows of data
        # Plot historical data
        fig = go.Figure() # Create a Plotly figure
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price')) # Add a line plot of Close Price
        fig.update_layout(title=f"{stock_ticker} Stock Prices", xaxis_title="Date", yaxis_title="Price") # Update layout
        st.plotly_chart(fig) # Display the Plotly figure
    else:
        st.error("No data found for the selected ticker and date range.")
else:
    st.warning("Please enter your Financial Modeling Prep API Key.")

# Prepare data for LSTM
if api_key and data is not None and not data.empty:
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) # Scale data between 0 and 1 

    # Create training dataset
    train_size = int(len(scaled_data) * 0.8) # 80% training data and 20% testing data
    train_data = scaled_data[:train_size] # Train data from start to 80%

    def create_dataset(dataset, time_step=60): # Create dataset for LSTM model with time steps for prediction 
        X, y = [], []
        for i in range(len(dataset) - time_step - 1): # Create X and y for each time step
            X.append(dataset[i:(i + time_step), 0]) # X is the historical data
            y.append(dataset[i + time_step, 0]) # y is the next day's price
        return np.array(X), np.array(y) # Convert to numpy array

    time_step = 60 # Number of time steps for prediction, default is 60 days
    X_train, y_train = create_dataset(train_data, time_step) # Create training dataset
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshape data for LSTM model input

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=(lstm_layers > 1), input_shape=(X_train.shape[1], 1))) # LSTM layer with 50 neurons, input shape is X_train, return sequences for multiple LSTM layers
    for _ in range(lstm_layers - 1): # Add additional LSTM layers if specified in the sidebar
        model.add(LSTM(50, return_sequences=False)) # LSTM layer with 50 neurons, return sequences for multiple LSTM layers
    model.add(Dense(1)) # Output layer with 1 neuron for predicting stock price

    model.compile(optimizer='adam', loss='mean_squared_error') # Compile model with Adam optimizer and mean squared error loss

    # Train model
    if st.sidebar.button("Train Model"):
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1) # Train model with specified epochs and batch size
        st.success("Model training complete!")

    # Predict future prices
    st.subheader("Predicted Stock Prices")
    test_data = scaled_data[train_size - time_step:]
    X_test, y_test = create_dataset(test_data, time_step)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # Reshape data for LSTM model input 

    predictions = model.predict(X_test) 
    predictions = scaler.inverse_transform(predictions) # Inverse transform scaled predictions to actual prices, for visualization
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)) # Inverse transform scaled actual prices to actual prices, for visualization

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=actual_prices.flatten(), mode='lines', name='Actual Prices')) # Plot actual prices
    fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted Prices')) # Plot predicted prices
    fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price") # Update layout
    st.plotly_chart(fig)

    # Download predicted data
    result_df = pd.DataFrame({
        "Date": data.index[-len(predictions):],
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predictions.flatten()
    })
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name=f"{stock_ticker}_predictions.csv", mime="text/csv") # Download button for predictions

# How it works section
with st.expander("How it Works"):
    st.write("This app fetches historical stock data from Financial Modeling Prep and trains a Long Short-Term Memory (LSTM) model to predict stock prices for the upcoming week. You can adjust parameters such as epochs, LSTM layers, and batch size to customize the prediction.")
    st.write("The predicted prices and historical prices are visualized interactively using Plotly.")
