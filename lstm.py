import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define LSTM model
def create_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 2), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(2)  # Predict future (x, y) positions
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and display the LSTM model
lstm_model = create_lstm_model()
lstm_model.summary()
