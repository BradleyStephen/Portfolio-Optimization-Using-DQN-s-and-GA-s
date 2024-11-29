import numpy as np
import pandas as pd
import random
from datetime import datetime

# Genetic Algorithm Parameters
POPULATION_SIZE = 500    # Keeping your optimized parameters
GENERATIONS = 50         # Keeping your optimized parameters
MUTATION_RATE = 0.1      # Keeping your optimized parameters
ELITISM_RATE = 0.05      # Maintains diversity

# Investment Parameters
BUDGET = 100000  # $100,000
RISK_FREE_RATE = 0.01  # 1% annual risk-free rate

# Portfolio Constraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.15  # Keeping your optimized parameters
REGULARIZATION_FACTOR = 0.2  # Increased L2 regularization factor
RETURN_RATIO_PENALTY_FACTOR = 50.0  # New penalty factor for return ratio
L1_REGULARIZATION_FACTOR = 0.02  # Increased L1 regularization to encourage sparsity
SHARPE_DIFF_PENALTY_FACTOR = 5.0  # Penalty factor for Sharpe ratio discrepancy

LATEST_COMMON_START_DATE = '2015-01-01'  # Align all datasets to start from this date

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
    
    # Handle missing data
    stocks.ffill(inplace=True)  # Forward-fill missing values
    stocks.dropna(inplace=True)  # Drop remaining NaN values

    # Print earliest date for debugging
    print(f"Earliest Date in Combined Data: {stocks['Date'].min()}")

    # Save the combined dataset to a CSV file for visualization
    stocks.to_csv('trimSet.csv', index=False)
    
    return stocks

def split_train_test(stocks_df):
    """Split data into training and testing sets based on date ranges"""
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    
    # Adjust date ranges based on data availability
    earliest_date = stocks_df['Date'].min()
    latest_date = stocks_df['Date'].max()
    print(f"Data available from {earliest_date} to {latest_date}")

    # Define date ranges
    test_start_date = '2021-07-01'
    
    # Split data
    train_df = stocks_df[stocks_df['Date'] < test_start_date]
    test_df = stocks_df[stocks_df['Date'] >= test_start_date]
    
    # Print data split info
    print(f"\nData Split:")
    print(f"Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Testing samples: {len(test_df)}")
    
    return train_df, test_df

def calculate_returns(df):
    """Calculate log returns for all stocks in the dataset"""
    stock_names = [col.split('_')[0] for col in df.columns if '_Close' in col]
    close_cols = [f"{stock}_Close" for stock in stock_names]
    prices = df[close_cols]
    returns_df = np.log(prices / prices.shift(1)).dropna()
    returns_df.columns = stock_names
    return returns_df, stock_names

def portfolio_metrics(weights, returns_df):
    """Calculate portfolio metrics including Sharpe ratio and annualized total return"""
    portfolio_returns = returns_df.dot(weights)
    mean_daily_return = portfolio_returns.mean()
    daily_return_std = portfolio_returns.std()
    
    if daily_return_std == 0:
        return 0, 0, 0, 0, 0, 0, 0  # Avoid division by zero
    
    # Annualize the returns
    annual_return = mean_daily_return * 252
    annual_std = daily_return_std * np.sqrt(252)
    
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_std
    
    # Calculate maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate total return over the entire period
    total_return = cum_returns.iloc[-1] - 1
    
    # Annualized total return
    num_days = len(portfolio_returns)
    num_years = num_days / 252  # Assuming 252 trading days per year
    annualized_total_return = (1 + total_return) ** (1 / num_years) - 1
    
    # Calculate Sortino Ratio
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - RISK_FREE_RATE) / downside_std if downside_std != 0 else 0
    
    # Calculate Calmar Ratio
    calmar_ratio = (annual_return - RISK_FREE_RATE) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return sharpe_ratio, annual_return, annual_std, max_drawdown, annualized_total_return, sortino_ratio, calmar_ratio

def rolling_window_cross_validation(returns_df, window_size, step_size):
    """Generator for rolling window cross-validation splits"""
    num_samples = len(returns_df)
    indices = np.arange(num_samples)
    max_start_index = num_samples - window_size - step_size

    for start in range(0, max_start_index + 1, step_size):
        train_indices = indices[start:start + window_size]
        val_indices = indices[start + window_size:start + window_size + step_size]
        if len(val_indices) == 0:
            break
        yield train_indices, val_indices

def portfolio_fitness_cv(weights, returns_df):
    """Fitness function that evaluates performance using rolling window cross-validation"""
    if not all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights) or abs(sum(weights) - 1) > 1e-6:
        return -np.inf

    window_size = int(0.5 * len(returns_df))  # Use 50% of data for training in each fold
    step_size = int(0.2 * len(returns_df))    # Use next 20% of data for validation
    sharpe_ratios = []

    for train_index, val_index in rolling_window_cross_validation(returns_df, window_size, step_size):
        train_returns_cv = returns_df.iloc[train_index]
        val_returns_cv = returns_df.iloc[val_index]

        # Calculate metrics on training and validation data
        sharpe_ratio_train, _, _, _, annualized_total_return_train, _, _ = portfolio_metrics(weights, train_returns_cv)
        sharpe_ratio_val, _, _, _, annualized_total_return_val, _, _ = portfolio_metrics(weights, val_returns_cv)
        
        # Calculate overfitting penalties
        return_ratio = (annualized_total_return_train / annualized_total_return_val) if annualized_total_return_val != 0 else np.inf
        if return_ratio > 1.2:  # If training return is more than 20% higher than validation return
            return_ratio_penalty = RETURN_RATIO_PENALTY_FACTOR * (return_ratio - 1.2)
        else:
            return_ratio_penalty = 0

        sharpe_diff_penalty = SHARPE_DIFF_PENALTY_FACTOR * max(0, (sharpe_ratio_train - sharpe_ratio_val))

        # Regularization terms
        l2_penalty = REGULARIZATION_FACTOR * np.sum(np.square(weights))
        l1_penalty = L1_REGULARIZATION_FACTOR * np.sum(np.abs(weights))

        # Fitness for this fold
        fitness = sharpe_ratio_val - l2_penalty - l1_penalty - return_ratio_penalty - sharpe_diff_penalty
        sharpe_ratios.append(fitness)

    # Average fitness across all folds
    avg_fitness = np.mean(sharpe_ratios)

    return avg_fitness

def initialize_population(pop_size, num_stocks):
    """Initialize random portfolio weights"""
    population = []
    while len(population) < pop_size:
        weights = np.random.dirichlet(np.ones(num_stocks))
        weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
        weights /= weights.sum()
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
    child = np.clip(child, MIN_WEIGHT, MAX_WEIGHT)
    child /= child.sum()  # Normalize
    return child

def mutate(child, mutation_rate):
    """Mutate child weights"""
    for i in range(len(child)):
        if random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.01)  # Reduced mutation magnitude for stability
            child[i] += mutation
    
    child = np.clip(child, MIN_WEIGHT, MAX_WEIGHT)
    if child.sum() == 0:
        child = np.ones_like(child) / len(child)
    else:
        child /= child.sum()
    return child

def genetic_algorithm_cv(returns_df, generations, pop_size, mutation_rate, elitism_rate):
    """Genetic algorithm function using cross-validation for fitness evaluation"""
    num_stocks = len(returns_df.columns)
    population = initialize_population(pop_size, num_stocks)
    best_weights = None
    best_fitness = -np.inf
    no_improvement_count = 0
    early_stopping_rounds = 20  # Increased for more exploration

    for generation in range(generations):
        # Evaluate fitness using cross-validation
        fitness_values = [portfolio_fitness_cv(ind, returns_df) for ind in population]

        # Sort population by fitness
        sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
        fitness_values_sorted = sorted(fitness_values, reverse=True)

        # Elitism
        elite_size = int(elitism_rate * pop_size)
        elites = sorted_population[:elite_size]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(sorted_population, fitness_values_sorted)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Keep track of best weights based on fitness
        current_best_fitness = fitness_values_sorted[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_weights = sorted_population[0]  # Corrected to select from sorted_population
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print("Early stopping triggered due to no improvement in fitness.")
            break

        if generation % 5 == 0 or generation == generations - 1:
            print(f"Generation {generation}, Best Fitness: {fitness_values_sorted[0]:.4f}")

    return best_weights

def evaluate_portfolio(weights, returns_df, stock_names, period_name=""):
    """Evaluate portfolio performance"""
    print(f"\n{period_name} Portfolio Evaluation:")
    
    sharpe_ratio, annual_return, annual_std, max_drawdown, annualized_total_return, sortino_ratio, calmar_ratio = portfolio_metrics(weights, returns_df)
    
    # Print allocation
    print("\nOptimal Allocation:")
    for stock, weight in zip(stock_names, weights):
        print(f"{stock}: {weight:.2%} (${BUDGET * weight:.2f})")
    
    # Print metrics
    print(f"\nExpected Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility (Std Dev): {annual_std:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Calmar Ratio: {calmar_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Annualized Total Return: {annualized_total_return:.2%}")
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'annual_return': annual_return,
        'annual_std_dev': annual_std,
        'max_drawdown': max_drawdown,
        'annualized_total_return': annualized_total_return
    }

def main():
    # Load data
    print("Loading and combining datasets...")
    stocks_df = load_and_combine_datasets()
    
    # Split into training and testing sets
    train_df, test_df = split_train_test(stocks_df)
    
    # Check if training data is empty
    if train_df.empty:
        raise ValueError("Training dataset is empty. Please adjust your date ranges or check your data.")

    # Calculate returns for each set
    returns_df, stock_names = calculate_returns(train_df)
    test_returns, _ = calculate_returns(test_df)
    
    # Run genetic algorithm with cross-validation on training data
    print("\nOptimizing portfolio weights using rolling window cross-validation...")
    best_weights = genetic_algorithm_cv(
        returns_df, GENERATIONS, POPULATION_SIZE, MUTATION_RATE, ELITISM_RATE
    )
    
    # Evaluate on training (full), and testing sets
    train_metrics = evaluate_portfolio(best_weights, returns_df, stock_names, "Training")
    test_metrics = evaluate_portfolio(best_weights, test_returns, stock_names, "Testing")
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Training Sharpe Ratio: {train_metrics['sharpe_ratio']:.4f}")
    print(f"Testing Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    print(f"Training Annualized Return: {train_metrics['annualized_total_return']:.2%}")
    print(f"Testing Annualized Return: {test_metrics['annualized_total_return']:.2%}")
    
    # Check for overfitting
    if train_metrics['annualized_total_return'] > 1.2 * test_metrics['annualized_total_return']:
        print("Warning: The training annualized return is significantly higher than the testing annualized return, indicating potential overfitting.")

if __name__ == "__main__":
    main()
