import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # To resolve potential OpenMP errors

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Hyperparameters for DQN
WINDOW_SIZE = 5  # Number of previous time steps to include
FEATURES_PER_STOCK = WINDOW_SIZE + 3  # Prices + Moving Average + Holdings + Cash
STATE_SIZE = FEATURES_PER_STOCK * 10  # 10 stocks
ACTION_SIZE = 5  # Actions: Hold, Buy 25%, Buy 50%, Sell 25%, Sell 50%
EPISODES = 50
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999  # Slower decay to allow more exploration
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 5

# Investment Parameters
BUDGET = 100000  # $100,000
RISK_FREE_RATE = 0.01  # 1% annual risk-free rate

# GA Optimal Allocations
GA_ALLOCATIONS = {
    'Amazon': 0.0420,
    'Apple': 0.1446,
    'Facebook': 0.0577,
    'Google': 0.1482,
    'Microsoft': 0.1449,
    'Netflix': 0.0552,
    'Tesla': 0.1496,
    'Uber': 0.1485,
    'Walmart': 0.1059,
    'Zoom': 0.0032
}

# Stock List
STOCKS = list(GA_ALLOCATIONS.keys())

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

    return stocks

def split_train_test(stocks_df):
    """Split data into training and testing sets based on an 80/20 split"""
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

    # Adjust date ranges based on data availability
    earliest_date = stocks_df['Date'].min()
    latest_date = stocks_df['Date'].max()
    print(f"Data available from {earliest_date} to {latest_date}")

    # Calculate the index to split at (80% of the data)
    split_index = int(0.8 * len(stocks_df))
    split_date = stocks_df['Date'].iloc[split_index]

    # Split data
    train_df = stocks_df[stocks_df['Date'] < split_date]
    test_df = stocks_df[stocks_df['Date'] >= split_date]

    # Print data split info
    print(f"\nData Split:")
    print(f"Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Testing samples: {len(test_df)}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

class TradingEnvironment:
    """Custom trading environment for the DQN agent"""

    def __init__(self, prices_df, initial_allocations):
        self.prices_df = prices_df.reset_index(drop=True)
        self.n_steps = len(prices_df)
        self.current_step = WINDOW_SIZE - 1  # Start from WINDOW_SIZE - 1
        self.done = False

        # Initial portfolio allocation based on GA
        self.initial_allocations = initial_allocations.copy()
        self.budget = BUDGET
        self.stock_names = STOCKS

        # Initialize holdings and cash
        self.holdings = {stock: 0 for stock in self.stock_names}
        self.cash = {stock: self.budget * self.initial_allocations[stock] for stock in self.stock_names}

        # Buy initial shares
        for stock in self.stock_names:
            price = self.prices_df.loc[self.current_step, f"{stock}_Close"]
            shares = self.cash[stock] // price
            self.holdings[stock] += shares
            self.cash[stock] -= shares * price

        # Track portfolio value over time
        self.portfolio_values = []

    def reset(self):
        self.current_step = WINDOW_SIZE - 1  # Reset to starting position
        self.done = False

        # Reset holdings and cash
        self.holdings = {stock: 0 for stock in self.stock_names}
        self.cash = {stock: self.budget * self.initial_allocations[stock] for stock in self.stock_names}

        # Buy initial shares
        for stock in self.stock_names:
            price = self.prices_df.loc[self.current_step, f"{stock}_Close"]
            shares = self.cash[stock] // price
            self.holdings[stock] += shares
            self.cash[stock] -= shares * price

        self.portfolio_values = []

        return self._get_state()

    def step(self, actions):
        """Take actions for each stock and return the next state, reward, and done flag"""

        # Check if we have reached the end
        if self.current_step >= self.n_steps - 1:
            self.done = True

        if self.done:
            # Return current state and reward of zero
            next_state = self._get_state()
            reward = 0
            return next_state, reward, self.done, {}

        # Ensure actions are valid
        # Actions: 0: Hold, 1: Buy 25%, 2: Buy 50%, 3: Sell 25%, 4: Sell 50%
        prices = {stock: self.prices_df.loc[self.current_step, f"{stock}_Close"] for stock in self.stock_names}
        total_reward = 0

        for idx, stock in enumerate(self.stock_names):
            action = actions[idx]
            price = prices[stock]

            if action == 1:  # Buy 25% of available cash
                cash_to_use = self.cash[stock] * 0.25
                shares_to_buy = cash_to_use // price
                if shares_to_buy > 0:
                    self.holdings[stock] += shares_to_buy
                    self.cash[stock] -= shares_to_buy * price

            elif action == 2:  # Buy 50% of available cash
                cash_to_use = self.cash[stock] * 0.50
                shares_to_buy = cash_to_use // price
                if shares_to_buy > 0:
                    self.holdings[stock] += shares_to_buy
                    self.cash[stock] -= shares_to_buy * price

            elif action == 3:  # Sell 25% of holdings
                shares_to_sell = int(self.holdings[stock] * 0.25)
                if shares_to_sell > 0:
                    self.holdings[stock] -= shares_to_sell
                    self.cash[stock] += shares_to_sell * price

            elif action == 4:  # Sell 50% of holdings
                shares_to_sell = int(self.holdings[stock] * 0.50)
                if shares_to_sell > 0:
                    self.holdings[stock] -= shares_to_sell
                    self.cash[stock] += shares_to_sell * price

            # Action 0 is Hold; do nothing

        # Calculate portfolio value
        portfolio_value = sum([
            self.holdings[stock] * self.prices_df.loc[self.current_step, f"{stock}_Close"] + self.cash[stock]
            for stock in self.stock_names
        ])
        self.portfolio_values.append(portfolio_value)

        # Reward: Change in portfolio value
        if len(self.portfolio_values) > 1:
            reward = self.portfolio_values[-1] - self.portfolio_values[-2]
        else:
            reward = 0

        total_reward += reward

        # Move to next step
        self.current_step += 1

        next_state = self._get_state()

        return next_state, reward, self.done, {}

    def _get_state(self):
        state = []
        # Ensure current_step does not exceed n_steps - 1
        current_step = min(self.current_step, self.n_steps - 1)
        for stock in self.stock_names:
            # Get price history
            start = current_step - WINDOW_SIZE + 1
            end = current_step + 1
            prices = self.prices_df.iloc[start:end][f"{stock}_Close"].values
            # Normalize prices
            prices = prices / prices[0] - 1  # Price returns over window
            # Moving average
            ma = prices.mean()
            # Holdings and cash (normalized)
            holdings = self.holdings[stock] / (self.budget / len(self.stock_names))
            cash = self.cash[stock] / (self.budget / len(self.stock_names))
            # Append features
            state.extend(prices.tolist())
            state.append(ma)
            state.append(holdings)
            state.append(cash)
        return np.array(state)

    def get_portfolio_performance(self):
        """Get portfolio returns for performance evaluation"""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        return returns

class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Total state size
        self.action_size = action_size  # Number of actions per stock
        self.num_stocks = len(STOCKS)
        self.memory = deque(maxlen=2000)
        self.gamma = GAMMA  # Discount rate
        self.epsilon = EPSILON_START  # Exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target model

    def _build_model(self):
        """Neural Network for Deep Q-Learning"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_stocks * self.action_size)
        )
        return model

    def update_target_model(self):
        """Update the target model to match the current model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """Return actions for the given state"""
        if np.random.rand() <= self.epsilon:
            # Random action
            actions = [np.random.choice(self.action_size) for _ in range(self.num_stocks)]
            return actions
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            q_values = q_values.view(self.num_stocks, self.action_size)
            actions = torch.argmax(q_values, dim=1).numpy()
            return actions

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train the model using experiences sampled from memory"""
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            action = torch.LongTensor(action)
            reward = torch.FloatTensor([reward])

            q_values = self.model(state).view(self.num_stocks, self.action_size)
            q_value = q_values[np.arange(self.num_stocks), action]

            with torch.no_grad():
                target_q_values = self.target_model(next_state).view(self.num_stocks, self.action_size)
                next_q_value = torch.max(target_q_values, dim=1)[0]
                expected_q_value = reward + (1 - done) * self.gamma * next_q_value

            loss = criterion(q_value, expected_q_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def evaluate_portfolio(returns_series, period_name=""):
    """Evaluate portfolio performance"""
    sharpe_ratio, annual_return, annual_std, max_drawdown, annualized_total_return = portfolio_metrics(returns_series)

    # Print metrics
    print(f"\n{period_name} Portfolio Evaluation:")
    print(f"Expected Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility (Std Dev): {annual_std:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Annualized Total Return: {annualized_total_return:.2%}")

    return {
        'sharpe_ratio': sharpe_ratio,
        'annual_return': annual_return,
        'annual_std_dev': annual_std,
        'max_drawdown': max_drawdown,
        'annualized_total_return': annualized_total_return
    }

def portfolio_metrics(returns_series):
    """Calculate portfolio metrics including Sharpe ratio and annualized total return"""
    mean_daily_return = returns_series.mean()
    daily_return_std = returns_series.std()

    if daily_return_std == 0:
        return 0, 0, 0, 0, 0  # Avoid division by zero

    # Annualize the returns
    annual_return = mean_daily_return * 252
    annual_std = daily_return_std * np.sqrt(252)

    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_std

    # Calculate maximum drawdown
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Calculate total return over the entire period
    total_return = cum_returns.iloc[-1] - 1

    # Number of trading days in the period
    num_days = len(returns_series)

    # Annualized total return adjusted for period length
    annualized_total_return = (1 + total_return) ** (252 / num_days) - 1

    return sharpe_ratio, annual_return, annual_std, max_drawdown, annualized_total_return

def main():
    # Load data
    print("Loading and combining datasets...")
    stocks_df = load_and_combine_datasets()

    # Split into training and testing sets using 80/20 split
    train_df, test_df = split_train_test(stocks_df)

    # Prepare prices for the environment
    close_cols = [f"{stock}_Close" for stock in STOCKS]
    train_prices = train_df[close_cols].reset_index(drop=True)
    test_prices = test_df[close_cols].reset_index(drop=True)

    # Initialize environment and agent
    initial_allocations = GA_ALLOCATIONS
    env = TradingEnvironment(train_prices, initial_allocations)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    # Training loop
    print("\nTraining DQN Agent...")
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = agent.act(state)
            next_state, reward, done, _ = env.step(actions)
            agent.remember(state, actions, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Agent replay
            agent.replay(BATCH_SIZE)

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        # Decay epsilon after each episode
        agent.decay_epsilon()

        print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Evaluate on training data
    train_returns = env.get_portfolio_performance()
    evaluate_portfolio(train_returns, "Training")

    # Testing phase
    print("\nTesting DQN Agent...")
    env_test = TradingEnvironment(test_prices, initial_allocations)
    state = env_test.reset()
    done = False

    while not done:
        actions = agent.act(state)
        next_state, reward, done, _ = env_test.step(actions)
        state = next_state

    # Evaluate on testing data
    test_returns = env_test.get_portfolio_performance()
    evaluate_portfolio(test_returns, "Testing")

if __name__ == "__main__":
    main()
