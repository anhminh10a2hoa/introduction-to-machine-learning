import gymnasium as gym
import numpy as np
import random

# Create Taxi environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 10000

# Initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Epsilon-greedy action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# Q-learning training
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

# Evaluation of the trained agent
def evaluate_agent(num_trials=10):
    total_rewards = []
    total_actions = []

    for trial in range(num_trials):
        state, _ = env.reset()
        done = False
        trial_reward = 0
        actions_count = 0

        while not done:
            action = np.argmax(Q[state])
            state, reward, done, truncated, info = env.step(action)
            trial_reward += reward
            actions_count += 1

        total_rewards.append(trial_reward)
        total_actions.append(actions_count)
    
    return np.mean(total_rewards), np.mean(total_actions)

# Print evaluation results
avg_reward, avg_actions = evaluate_agent()
print(f"Average Reward after training: {avg_reward}")
print(f"Average number of actions: {avg_actions}")

# Save the Q-table
np.save("taxi_q_table.npy", Q)
