import numpy as np
import pandas as pd
from functools import reduce

files = ['datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv', 'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv', 'datasets/Tesla.csv', 'datasets/Uber.csv', 'datasets/Walmart.csv', 'datasets/Zoom.csv']
dfs = []

for file in files:
    # Read each file
    temp = pd.read_csv(file)
    
    # Rename the columns to include the stock name (except 'Date')
    stock_name = file.replace('.csv', '')
    temp.columns = ['Date'] + [f"{stock_name}_{col}" for col in temp.columns[1:]]
    
    # Append to the list of DataFrames
    dfs.append(temp)

# Merge all dataframes on 'Date'
stocks = reduce(lambda left, right: pd.merge(left, right, on='Date'), dfs)
stocks.columns = stocks.columns.str.replace('datasets/', '')

# Check the shape and view the updated DataFrame
print(stocks.shape)
print(stocks.head(10))  # Display the first 10 rows

print(stocks.head(10))  # Display the first 10 rows

