import numpy as np
import pandas as pd
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt  # Added for plotting

# Portfolio and DQN Parameters
BUDGET = 100000
MAX_EPISODES = 50  # Adjust as needed for training
MAX_STEPS = 100  # Maximum steps per episode
GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 0.001
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
DELTA_WEIGHT = 0.05  # Weight adjustment step size
RISK_FREE_RATE = 0.01 / 252  # Daily risk-free rate (1% annualized)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def load_and_combine_datasets():
    """Load and combine stock datasets."""
    files = [
        'datasets/Amazon.csv', 'datasets/Apple.csv', 'datasets/Facebook.csv',
        'datasets/Google.csv', 'datasets/Microsoft.csv', 'datasets/Netflix.csv',
        'datasets/Tesla.csv', 'datasets/Walmart.csv'
    ]
    dfs = []

    for file in files:
        temp = pd.read_csv(file)
        stock_name = file.replace('datasets/', '').replace('.csv', '')
        temp.columns = ['Date'] + [f"{stock_name}_{col}" for col in temp.columns[1:]]
        temp['Date'] = pd.to_datetime(temp['Date'])
        temp = temp[temp['Date'] >= '2015-05-10']  # Aligning with the earliest common start date
        dfs.append(temp)

    # Merge all dataframes on 'Date'
    stocks = dfs[0]
    for df in dfs[1:]:
        stocks = pd.merge(stocks, df, on='Date', how='inner')

    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks = stocks.sort_values('Date').reset_index(drop=True)
    stocks.dropna(inplace=True)

    return stocks

def calculate_returns(stocks_df):
    """Calculate returns for all stocks in the dataset."""
    stock_names = [col.split('_')[0] for col in stocks_df.columns if '_Close' in col]
    close_cols = [f"{stock}_Close" for stock in stock_names]
    returns_df = stocks_df[close_cols].pct_change().dropna()
    returns_df.columns = stock_names
    return returns_df, stock_names

class PortfolioEnv:
    """Portfolio optimization environment."""
    def __init__(self, returns_df, initial_weights):
        self.returns_df = returns_df.reset_index(drop=True)
        self.n_assets = returns_df.shape[1]
        self.current_step = 0
        self.action_size = self.n_assets * 2  # Increase or decrease each asset
        self.delta_weight = DELTA_WEIGHT
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        initial_weights = np.ones(self.n_assets) / self.n_assets
        initial_returns = np.zeros(5)  # Fixed-length history of returns
        self.state = np.hstack([initial_weights, initial_returns])
        return self.state

    def step(self, action):
        """Take an action and return the new state, reward, and done flag."""
        weights = self.state[:self.n_assets]
        returns = self.state[self.n_assets:]

        # Map action index to weight adjustment
        asset_index = action % self.n_assets
        adjustment = self.delta_weight if action < self.n_assets else -self.delta_weight
        weight_adjustments = np.zeros(self.n_assets)
        weight_adjustments[asset_index] = adjustment

        # Apply the weight adjustments
        new_weights = weights + weight_adjustments
        new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
        if new_weights.sum() == 0:
            new_weights = np.ones(self.n_assets) / self.n_assets  # Reset to equal weights if sum is zero
        else:
            new_weights /= new_weights.sum()  # Normalize weights to sum to 1

        # Simulate portfolio returns
        portfolio_return = np.dot(new_weights, self.returns_df.iloc[self.current_step])

        # Update returns history
        returns = np.roll(returns, -1)
        returns[-1] = portfolio_return

        # Update state
        self.state = np.hstack([new_weights, returns])

        # Reward: Sharpe ratio proxy (simplified)
        if np.std(returns) == 0:
            reward = 0
        else:
            reward = (portfolio_return - RISK_FREE_RATE) / np.std(returns)

        self.current_step += 1
        done = self.current_step >= len(self.returns_df) - 1
        return self.state, reward, done

class DQNAgent:
    """Deep Q-Network agent."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Build the Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def update_target_model(self):
        """Update target network weights."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using an epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            # Exploration: Random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: Predict Q-values and take the action with the highest value
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            return np.argmax(q_values[0])

    def replay(self):
        """Train the model using experiences from the replay buffer."""
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict current Q-values
        q_values = self.model.predict(states, verbose=0)
        # Predict next Q-values using target network
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * np.amax(q_next[i])
            q_values[i][actions[i]] = target

        # Train the model
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def train_dqn(returns_df, stock_names, risk_free_rate=RISK_FREE_RATE):
    """Train the DQN agent."""
    initial_weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    env = PortfolioEnv(returns_df, initial_weights)
    agent = DQNAgent(state_size=env.state.size, action_size=env.action_size)

    # Initialize logging variables
    cumulative_rewards = []
    sharpe_ratios = []
    max_drawdowns = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_returns = []
        portfolio_values = [1.0]  # Starting portfolio value (normalized)

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Track returns for cumulative performance metrics
            portfolio_return = state[env.n_assets:][-1]  # Latest portfolio return
            episode_returns.append(portfolio_return)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            if done:
                break

        # Replay and update the agent
        agent.replay()
        agent.decay_epsilon()
        agent.update_target_model()

        # Calculate episode metrics
        cumulative_return = portfolio_values[-1] - 1
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0
        max_drawdown = np.min((np.array(portfolio_values) - np.maximum.accumulate(portfolio_values)) / np.maximum.accumulate(portfolio_values))

        # Store metrics
        cumulative_rewards.append(cumulative_return)
        sharpe_ratios.append(sharpe_ratio)
        max_drawdowns.append(max_drawdown)

        # Log episode summary
        final_weights = state[:env.n_assets]
        # Include stock names next to weights
        weight_info = ', '.join([f"{name}: {weight:.4f}" for name, weight in zip(stock_names, final_weights)])
        print(f"Episode {episode + 1}/{MAX_EPISODES}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Cumulative Return: {cumulative_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Final Portfolio Weights:")
        print(f"    {weight_info}")
        print(f"  Epsilon (Exploration Rate): {agent.epsilon:.2f}\n")

    # Summary of training performance
    print("\nTraining Complete!")
    print("Final Results:")
    print(f"  Average Cumulative Return: {np.mean(cumulative_rewards):.2%}")
    print(f"  Average Sharpe Ratio: {np.mean(sharpe_ratios):.4f}")
    print(f"  Average Max Drawdown: {np.mean(max_drawdowns):.2%}")

    # Plotting metrics over episodes
    episodes = range(1, MAX_EPISODES + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, cumulative_rewards, label='Cumulative Return')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return over Episodes')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(episodes, sharpe_ratios, label='Sharpe Ratio', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio over Episodes')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(episodes, max_drawdowns, label='Max Drawdown', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Max Drawdown')
    plt.title('Max Drawdown over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess stock data
    print("Loading and combining datasets...")
    stocks_df = load_and_combine_datasets()

    # Calculate daily returns
    returns_df, stock_names = calculate_returns(stocks_df)

    # Train the DQN agent
    print("\nTraining the DQN agent...")
    train_dqn(returns_df, stock_names)

if __name__ == "__main__":
    main()
