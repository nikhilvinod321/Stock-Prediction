import pandas as pd
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

stock_data = pd.read_csv('stock_data.csv', header=2)
print(stock_data.columns)

X, _, _ = preprocess_data(stock_data)

if isinstance(X, pd.DataFrame):
    plt.plot(X.iloc[:, 0])
else:
    plt.plot(X[:, 0])

plt.title('First Feature Visualization')
plt.show()
