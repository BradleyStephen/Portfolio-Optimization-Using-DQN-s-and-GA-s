import numpy as np
import pandas as pd
import random

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1

# Investment Parameters
BUDGET = 100000  # $100,000
RISK_FREE_RATE = 0.01 / 252  # Approximate daily risk-free rate (assuming 1% annualized risk-free rate)

# Constraints (minimum and maximum allocation per stock)
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.4  # Maximum 40% allocation to a single stock

# Load and Combine Datasets
def load_and_combine_datasets():
    files = [
        'datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv',
        'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv',
        'datasets/Tesla.csv', 'datasets/Uber.csv', 'datasets/Walmart.csv', 'datasets/Zoom.csv'
    ]
    dfs = []

    for file in files:
        # Read each file
        temp = pd.read_csv(file)

        # Rename the columns to include the stock name (except 'Date')
        stock_name = file.replace('datasets/', '').replace('.csv', '')
        temp.columns = ['Date'] + ["{}_{}".format(stock_name, col) for col in temp.columns[1:]]

        # Append to the list of DataFrames
        dfs.append(temp)

    # Merge all dataframes on 'Date'
    stocks = dfs[0]
    for df in dfs[1:]:
        stocks = pd.merge(stocks, df, on='Date', how='inner')

    # Convert 'Date' to datetime and sort
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks = stocks.sort_values('Date').reset_index(drop=True)

    # Drop any rows with missing values
    stocks.dropna(inplace=True)

    # Save the combined dataset to 'stocks.csv'
    stocks.to_csv('stocks.csv', index=False)
    print("Combined dataset saved to 'stocks.csv'.")

    return stocks

# Objective Function: Calculate Sharpe Ratio of the Portfolio
def portfolio_sharpe_ratio(weights, returns_df):
    # Enforce constraints
    if not all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights):
        return -np.inf  # Penalize portfolios violating the constraints

    if abs(np.sum(weights) - 1) > 1e-6:
        return -np.inf  # Penalize portfolios that do not sum up to 1

    # Calculate portfolio returns
    portfolio_returns = returns_df.dot(weights)
    excess_returns = portfolio_returns - RISK_FREE_RATE
    mean_excess_return = excess_returns.mean()
    return_std = portfolio_returns.std()
    if return_std == 0:
        return 0  # Avoid division by zero
    sharpe_ratio = mean_excess_return / return_std
    return sharpe_ratio

# Initialize Population (Random Portfolio Weights)
def initialize_population(pop_size, num_stocks):
    population = []
    while len(population) < pop_size:
        weights = np.random.dirichlet(np.ones(num_stocks), size=1)[0]
        # Enforce constraints
        if all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights):
            population.append(weights)
    return population

# Select Parents (Tournament Selection)
def select_parents(population, fitness_values, tournament_size=3):
    selected = []
    for _ in range(2):
        participants = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover (Blend the Weights of Two Parents)
def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child = alpha * parent1 + (1 - alpha) * parent2
    child /= np.sum(child)  # Normalize to ensure sum is 1
    return child

# Mutate (Random Adjustment of Weights)
def mutate(child, mutation_rate):
    for i in range(len(child)):
        if random.uniform(0, 1) < mutation_rate:
            mutation_value = np.random.normal(0, 0.05)  # Smaller standard deviation for finer adjustments
            child[i] += mutation_value
    child = np.clip(child, MIN_WEIGHT, MAX_WEIGHT)  # Ensure weights stay within constraints
    if child.sum() == 0:
        child = np.ones_like(child) / len(child)  # Reinitialize if all weights are zero
    else:
        child /= np.sum(child)  # Re-normalize to ensure sum is 1
    return child

# Get stock names
def get_stock_names(stocks_df):
    return [col.split('_')[0] for col in stocks_df.columns if '_Close' in col]

# Calculate Extended Metrics
def calculate_extended_metrics(portfolio_returns):
    # Sortino Ratio
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = negative_returns.std(ddof=1)
    sortino_ratio = portfolio_returns.mean() / downside_std if downside_std != 0 else 0

    # Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = (portfolio_returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0

    # Value at Risk (VaR) at 95% confidence level
    var_95 = np.percentile(portfolio_returns, 5)

    return sortino_ratio, max_drawdown, calmar_ratio, var_95

# Main Genetic Algorithm Function
def genetic_algorithm(stocks_df, generations, pop_size, mutation_rate):
    stock_names = get_stock_names(stocks_df)
    # Extract the closing prices and calculate daily returns
    close_price_cols = ["{}_Close".format(stock) for stock in stock_names]
    returns_df = stocks_df[close_price_cols].pct_change().dropna()
    returns_df.columns = stock_names  # Rename columns to stock names
    num_stocks = len(stock_names)

    # Initialize Population
    population = initialize_population(pop_size, num_stocks)

    for generation in range(generations):
        # Evaluate Fitness of Each Individual
        fitness_values = [portfolio_sharpe_ratio(individual, returns_df) for individual in population]

        # Create New Population
        new_population = []
        while len(new_population) < pop_size:
            # Select Parents
            parent1, parent2 = select_parents(population, fitness_values)
            # Perform Crossover
            child = crossover(parent1, parent2)
            # Perform Mutation
            child = mutate(child, mutation_rate)
            # Add Child to New Population
            new_population.append(child)

        population = new_population

        # Print Best Fitness in Current Generation
        best_fitness = max(fitness_values)
        print("Generation {}, Best Fitness (Sharpe Ratio): {:.4f}".format(generation + 1, best_fitness))

    # Get the Best Individual from the Final Generation
    final_fitness_values = [portfolio_sharpe_ratio(individual, returns_df) for individual in population]
    best_index = np.argmax(final_fitness_values)
    best_individual = population[best_index]

    # Print Optimal Allocation
    allocation = {stock: weight for stock, weight in zip(stock_names, best_individual)}
    print("\nOptimal Allocation:")
    for stock, weight in allocation.items():
        print("{}: {:.2%} (${:.2f})".format(stock, weight, weight * BUDGET))

    # Calculate and Print Expected Portfolio Return and Risk
    portfolio_returns = returns_df.dot(best_individual)
    excess_returns = portfolio_returns - RISK_FREE_RATE
    mean_return = portfolio_returns.mean()
    return_std = portfolio_returns.std()
    sharpe_ratio = (mean_return - RISK_FREE_RATE) / return_std if return_std != 0 else 0

    # Calculate Extended Metrics
    sortino_ratio, max_drawdown, calmar_ratio, var_95 = calculate_extended_metrics(portfolio_returns)

    print("\nExpected Daily Portfolio Return: {:.4%}".format(mean_return))
    print("Expected Annualized Return: {:.2%}".format(mean_return * 252))
    print("Portfolio Standard Deviation (Daily): {:.4%}".format(return_std))
    print("Portfolio Sharpe Ratio: {:.4f}".format(sharpe_ratio))
    print("Portfolio Sortino Ratio: {:.4f}".format(sortino_ratio))
    print("Maximum Drawdown: {:.2%}".format(max_drawdown))
    print("Calmar Ratio: {:.4f}".format(calmar_ratio))
    print("Value at Risk (95% confidence): {:.2%}".format(var_95))

    # Expected Profit/Loss over the investment period
    total_days = len(portfolio_returns)
    total_return = (1 + portfolio_returns).prod() - 1
    expected_profit = total_return * BUDGET
    print("\nTotal Expected Return over {} days: {:.2%}".format(total_days, total_return))
    print("Expected Profit/Loss: ${:.2f}".format(expected_profit))

# Load and Combine Datasets, then Save to 'stocks.csv'
stocks = load_and_combine_datasets()

# Alternatively, read from 'stocks.csv' if it already exists
# stocks = pd.read_csv('stocks.csv')

# Run Genetic Algorithm using the combined dataset
genetic_algorithm(stocks, GENERATIONS, POPULATION_SIZE, MUTATION_RATE)
