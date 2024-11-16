import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def predict_stock_price(model, X, scaler):
    predicted_price = model.predict(X)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

if __name__ == "__main__":
    model = tf.keras.models.load_model('stock_prediction_model.h5')
    stock_data = pd.read_csv('stock_data.csv')
    test_data = stock_data['Close'][-60:].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test_data = scaler.fit_transform(test_data)
    X_test = []
    X_test.append(scaled_test_data)
    X_test = np.array(X_test)
    predicted_price = predict_stock_price(model, X_test, scaler)
    print(f"Predicted stock price: {predicted_price[0][0]}")
