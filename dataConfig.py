import numpy as np
import pandas as pd
import random
from functools import reduce

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1

# Load Dataset
files = ['datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv', 
         'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv', 
         'datasets/Tesla.csv', 'datasets/Uber.csv', 'datasets/Walmart.csv', 'datasets/Zoom.csv']
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
stocks = reduce(lambda left, right: pd.merge(left, right, on='Date'), dfs)

stocks = stocks.sort_values('Date', ascending=False).reset_index(drop=True)

# Objective Function: Calculate Portfolio Return
def portfolio_return(weights, returns_df):
    portfolio_returns = (weights * returns_df).sum(axis=1)
    return portfolio_returns.mean()

# Initialize Population (Random Portfolio Weights)
def initialize_population(pop_size, num_stocks):
    return [np.random.dirichlet(np.ones(num_stocks), size=1)[0] for _ in range(pop_size)]

# Select Parents (Roulette Wheel Selection)
def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [fitness / total_fitness for fitness in fitness_values]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

# Crossover (Blend the Weights of Two Parents)
def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child = alpha * parent1 + (1 - alpha) * parent2
    return child

# Mutate (Random Adjustment of Weights)
def mutate(child, mutation_rate):
    if random.uniform(0, 1) < mutation_rate:
        mutation_idx = random.randint(0, len(child) - 1)
        mutation_value = random.uniform(-0.1, 0.1)
        child[mutation_idx] += mutation_value
        child = np.clip(child, 0, 1)  # Ensure weights stay between 0 and 1
        child /= np.sum(child)  # Re-normalize to ensure sum is 1
    return child

# Get stock names
def get_stock_names(stocks_df):
    return [col.split('_')[0] for col in stocks_df.columns if '_Close' in col]

# Main Genetic Algorithm Function
def genetic_algorithm(stocks_df, generations, pop_size, mutation_rate):
    stock_names = get_stock_names(stocks_df)
    returns_df = stocks_df[["{}_Close".format(stock) for stock in stock_names]].pct_change().dropna()
    num_stocks = len(stock_names)

    # Initialize Population
    population = initialize_population(pop_size, num_stocks)

    for generation in range(generations):
        # Evaluate Fitness of Each Individual
        fitness_values = [portfolio_return(individual, returns_df) for individual in population]

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
        print("Generation {}, Best Fitness: {:.4f}".format(generation + 1, best_fitness))

    # Get the Best Individual from the Final Generation
    final_fitness_values = [portfolio_return(individual, returns_df) for individual in population]
    best_individual = population[np.argmax(final_fitness_values)]

    # Print Optimal Allocation
    allocation = {stock: weight for stock, weight in zip(stock_names, best_individual)}
    print("\nOptimal Allocation:")
    for stock, weight in allocation.items():
        print("{}: {:.2%}".format(stock, weight))

    # Calculate and Print Expected Portfolio Return
    expected_return = portfolio_return(best_individual, returns_df)
    print("\nExpected Portfolio Return: {:.4f}".format(expected_return))

# Run Genetic Algorithm
genetic_algorithm(stocks, GENERATIONS, POPULATION_SIZE, MUTATION_RATE)