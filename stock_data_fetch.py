import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

if __name__ == "__main__":
    stock_data = fetch_stock_data()
    stock_data.to_csv('stock_data.csv')
    print("Stock data fetched and saved as stock_data.csv")
