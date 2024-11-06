import numpy as np
import pandas as pd
import random
from datetime import datetime

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 30  # Further reduced to prevent overfitting
MUTATION_RATE = 0.4  # Further increased to introduce more randomness and avoid overfitting

# Investment Parameters
BUDGET = 100000  # $100,000
RISK_FREE_RATE = 0.01 / 252  # Approximate daily risk-free rate (1% annual)

# Portfolio Constraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.15  # Further reduced maximum allocation per stock to encourage greater diversification
REGULARIZATION_FACTOR = 0.15  # Reduced penalty to balance exploration and diversification
DROPOUT_RATE = 0.3  # Increased dropout rate for asset selection during training

LATEST_COMMON_START_DATE = '2019-05-10'  # Align all datasets to start from this date

def load_and_combine_datasets():
    """Load and combine stock datasets"""
    files = [
        'datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv',
        'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv',
        'datasets/Tesla.csv', 'datasets/Uber.csv', 'datasets/Walmart.csv', 'datasets/Zoom.csv'
    ]
    dfs = []

    for file in files:
        temp = pd.read_csv(file)
        stock_name = file.replace('datasets/', '').replace('.csv', '')
        temp.columns = ['Date'] + [f"{stock_name}_{col}" for col in temp.columns[1:]]
        temp['Date'] = pd.to_datetime(temp['Date'])
        temp = temp[temp['Date'] >= LATEST_COMMON_START_DATE]  # Filter to the latest common start date
        dfs.append(temp)

    # Merge all dataframes on 'Date'
    stocks = dfs[0]
    for df in dfs[1:]:
        stocks = pd.merge(stocks, df, on='Date', how='inner')

    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks = stocks.sort_values('Date').reset_index(drop=True)
    stocks.dropna(inplace=True)

    # Save the combined dataset to a CSV file for visualization
    stocks.to_csv('trimSet.csv', index=False)
    
    return stocks

def split_train_validation_test(stocks_df, train_ratio=0.6, validation_ratio=0.2):
    """Split data into training, validation, and testing sets"""
    train_end_idx = int(len(stocks_df) * train_ratio)
    validation_end_idx = int(len(stocks_df) * (train_ratio + validation_ratio))
    
    train_df = stocks_df.iloc[:train_end_idx]
    validation_df = stocks_df.iloc[train_end_idx:validation_end_idx]
    test_df = stocks_df.iloc[validation_end_idx:]
    
    print(f"\nData Split:")
    print(f"Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Validation period: {validation_df['Date'].min()} to {validation_df['Date'].max()}")
    print(f"Testing period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Training samples: {len(train_df)}, Validation samples: {len(validation_df)}, Testing samples: {len(test_df)}")
    
    return train_df, validation_df, test_df

def calculate_returns(df):
    """Calculate returns for all stocks in the dataset"""
    stock_names = [col.split('_')[0] for col in df.columns if '_Close' in col]
    close_cols = [f"{stock}_Close" for stock in stock_names]
    returns_df = df[close_cols].pct_change().dropna()
    returns_df.columns = stock_names
    return returns_df, stock_names

def portfolio_metrics(weights, returns_df):
    """Calculate portfolio metrics including Sharpe ratio"""
    portfolio_returns = returns_df.dot(weights)
    mean_return = portfolio_returns.mean()
    return_std = portfolio_returns.std()
    
    if return_std == 0:
        return 0, 0, 0, 0, 0  # Avoid division by zero
    
    sharpe_ratio = (mean_return - RISK_FREE_RATE) / return_std
    
    # Annualize the Sharpe ratio
    annual_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    
    # Calculate maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate total return over the entire period
    total_return = cum_returns.iloc[-1] - 1
    
    return annual_sharpe_ratio, mean_return, return_std, max_drawdown, total_return

def portfolio_sharpe_ratio(weights, returns_df):
    """Fitness function for genetic algorithm, with regularization to encourage diversification"""
    if not all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights) or abs(sum(weights) - 1) > 1e-6:
        return -np.inf
    
    sharpe_ratio, _, _, _, _ = portfolio_metrics(weights, returns_df)
    regularization_penalty = REGULARIZATION_FACTOR * np.sum(weights ** 2)
    return sharpe_ratio - regularization_penalty

def initialize_population(pop_size, num_stocks):
    """Initialize random portfolio weights"""
    population = []
    while len(population) < pop_size:
        weights = np.random.dirichlet(np.ones(num_stocks))
        if all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights):
            population.append(weights)
    return population

def select_parents(population, fitness_values, tournament_size=3):
    """Select parents using tournament selection"""
    selected = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    alpha = random.random()
    child = alpha * parent1 + (1 - alpha) * parent2
    child /= child.sum()  # Normalize
    return child

def mutate(child, mutation_rate):
    """Mutate child weights"""
    for i in range(len(child)):
        if random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.05)
            child[i] += mutation
    
    child = np.clip(child, MIN_WEIGHT, MAX_WEIGHT)
    if child.sum() == 0:
        child = np.ones_like(child) / len(child)
    else:
        child /= child.sum()
    return child

def apply_dropout(weights, dropout_rate):
    """Apply dropout to the portfolio weights to encourage diversification"""
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=len(weights))
    weights = weights * dropout_mask
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights.sum()
    return weights

def genetic_algorithm(train_returns, validation_returns, generations, pop_size, mutation_rate):
    """Main genetic algorithm function with early stopping and dropout"""
    num_stocks = len(train_returns.columns)
    population = initialize_population(pop_size, num_stocks)
    best_fitness_history = []
    best_weights = None
    best_validation_fitness = -np.inf
    early_stopping_rounds = 15  # Increased to make early stopping less sensitive
    no_improvement_count = 0
    
    for generation in range(generations):
        fitness_values = [portfolio_sharpe_ratio(ind, train_returns) for ind in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        
        if generation % 10 == 0:
            print(f"Generation {generation}, Best Fitness: {best_fitness:.4f}")
        
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitness_values)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            child = apply_dropout(child, DROPOUT_RATE)  # Apply dropout to encourage diversification
            new_population.append(child)
        
        population = new_population
        
        # Early stopping based on validation fitness
        validation_fitness_values = [portfolio_sharpe_ratio(ind, validation_returns) for ind in population]
        current_best_validation_fitness = max(validation_fitness_values)
        if current_best_validation_fitness > best_validation_fitness:
            best_validation_fitness = current_best_validation_fitness
            best_weights = population[np.argmax(validation_fitness_values)]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stopping_rounds:
            print("Early stopping triggered due to no improvement in validation fitness.")
            break
    
    return best_weights, best_fitness_history

def evaluate_portfolio(weights, returns_df, stock_names, period_name=""):
    """Evaluate portfolio performance"""
    print(f"\n{period_name} Portfolio Evaluation:")
    
    # Calculate metrics
    sharpe_ratio, mean_return, return_std, max_drawdown, total_return = portfolio_metrics(weights, returns_df)
    
    # Annualize the expected return
    annual_return = mean_return * 252
    
    # Print allocation
    print("\nOptimal Allocation:")
    for stock, weight in zip(stock_names, weights):
        print(f"{stock}: {weight:.2%} (${BUDGET * weight:.2f})")
    
    # Print metrics
    print(f"\nExpected Daily Return: {mean_return:.4%}")
    print(f"Expected Annual Return: {annual_return:.2%}")
    print(f"Portfolio Standard Deviation (Daily): {return_std:.4%}")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Total Period Return: {total_return:.2%}")
    print(f"Total Profit/Loss: ${BUDGET * total_return:.2f}")
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'mean_return': mean_return,
        'std_dev': return_std,
        'max_drawdown': max_drawdown,
        'total_return': total_return
    }

def main():
    # Load data
    print("Loading and combining datasets...")
    stocks_df = load_and_combine_datasets()
    
    # Split into training, validation, and testing sets
    train_df, validation_df, test_df = split_train_validation_test(stocks_df, train_ratio=0.6, validation_ratio=0.2)
    
    # Calculate returns for each set
    train_returns, stock_names = calculate_returns(train_df)
    validation_returns, _ = calculate_returns(validation_df)
    test_returns, _ = calculate_returns(test_df)
    
    # Run genetic algorithm on training data with validation
    print("\nOptimizing portfolio weights using training and validation data...")
    best_weights, fitness_history = genetic_algorithm(
        train_returns, validation_returns, GENERATIONS, POPULATION_SIZE, MUTATION_RATE
    )
    
    # Evaluate on training, validation, and testing sets
    train_metrics = evaluate_portfolio(best_weights, train_returns, stock_names, "Training")
    validation_metrics = evaluate_portfolio(best_weights, validation_returns, stock_names, "Validation")
    test_metrics = evaluate_portfolio(best_weights, test_returns, stock_names, "Testing")
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Training Sharpe Ratio: {train_metrics['sharpe_ratio']:.4f}")
    print(f"Validation Sharpe Ratio: {validation_metrics['sharpe_ratio']:.4f}")
    print(f"Testing Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"Training Return: {train_metrics['total_return']:.2%}")
    print(f"Validation Return: {validation_metrics['total_return']:.2%}")

    print(f"Testing Return: {test_metrics['total_return']:.2%}")

    # Check for overfitting
    if train_metrics['sharpe_ratio'] > 2 * test_metrics['sharpe_ratio']:
        print("\nWarning: The model may be overfitting to the training data. Consider adjusting the Genetic Algorithm parameters or using more data.")
    if train_metrics['total_return'] > 5 * test_metrics['total_return']:
        print("Warning: The training return is significantly higher than the testing return, indicating potential overfitting.")

if __name__ == "__main__":
    main()