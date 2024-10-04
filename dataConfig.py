import numpy as np
import pandas as pd
from functools import reduce

files = ['datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv', 
         'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv', 
         'datasets/Tesla.csv', 'datasets/Uber.csv', 'datasets/Walmart.csv', 'datasets/Zoom.csv']
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

stocks = stocks.sort_values('Date', ascending=False).reset_index(drop=True)

# Print dataframe info
print(f"Dataframe shape: {stocks.shape}")
print(f"Date range: {stocks['Date'].iloc[-1]} to {stocks['Date'].iloc[0]}")
print("\nFirst few columns:")
print(list(stocks.columns)[:5])

def calculate_historical_returns(stocks_df, periods):
    results = {}
    stock_names = [col.split('_')[0] for col in stocks_df.columns if '_Close' in col]
    
    print("\nCalculating returns for stocks:", stock_names)
    
    for stock in stock_names:
        stock_close = f"{stock}_Close"
        
        if stock_close not in stocks_df.columns:
            print(f"Warning: {stock_close} not found in dataframe")
            continue
        
        stock_returns = {}
        print(f"\n{stock} prices:")
        print(f"Current ({stocks_df['Date'].iloc[0]}): ${stocks_df[stock_close].iloc[0]:.2f}")
        
        for period in periods:
            if period < len(stocks_df):
                recent_price = stocks_df[stock_close].iloc[0]
                old_price = stocks_df[stock_close].iloc[period]
                date = stocks_df['Date'].iloc[period]
                
                print(f"{period//21}m ago ({date}): ${old_price:.2f}")
                
                period_return = (recent_price - old_price) / old_price
                months = period // 21
                stock_returns[f"{months}m"] = period_return
        
        results[stock] = stock_returns
    
    return pd.DataFrame(results).T

# Define periods
periods = [
    21 * 3,   # 3 months
    21 * 6,   # 6 months
    21 * 12,  # 12 months
    21 * 24   # 24 months
]

# Calculate returns
historical_returns = calculate_historical_returns(stocks, periods)

# Format as percentages
formatted_returns = historical_returns.apply(lambda x: x.map('{:.2%}'.format))

print("\nHistorical Returns:")
print(formatted_returns)