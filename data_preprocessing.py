import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_close = stock_data[['Unnamed: 1']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_stock_data = scaler.fit_transform(stock_close)
    X, y = [], []
    time_step = 60
    for i in range(time_step, len(scaled_stock_data)):
        X.append(scaled_stock_data[i-time_step:i, 0])
        y.append(scaled_stock_data[i, 0])
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X, y, scaler

stock_data = pd.read_csv('stock_data.csv', header=2)
print(stock_data.columns)
print(stock_data.head())

X, y, scaler = preprocess_data(stock_data)
X.to_csv('X_processed.csv', index=False)
y.to_csv('y_processed.csv', index=False)
print(f'Processed data shapes: X={X.shape}, y={y.shape}')

